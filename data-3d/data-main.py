import os
import subprocess
import numpy as np
import re
import math
import logging
import time
import h5py

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting main.py script")

# 配置路径
working_directory = "/public/home/hpc221253/specfem3d/specfem3d-master/EXAMPLES/30-500menyuan"
cmtsolution_path = os.path.join(working_directory, "DATA", "CMTSOLUTION")
specfem_sh_path = os.path.join(working_directory, "specfem.sh")
output_files_dir = os.path.join(working_directory, "OUTPUT_FILES")
output_hdf5_path = "/public/home/hpc221253/pytorch_gpu/specfem3d-py/data_3d"
source_positions_path = os.path.join(output_hdf5_path, "source_positions.txt")
transformed_positions_path = os.path.join(output_hdf5_path, "transformed_positions.txt")

# 修改CMTSOLUTION文件
def modify_cmtsolution(new_source_x, new_source_z):
    with open(cmtsolution_path, 'r') as file:
        content = file.read()

    new_source_x1 = new_source_x / 64  * 20000+720000
    new_source_z1 = -new_source_z / 64 *10000 -3000
    new_source_y1 = -0.745 *  new_source_x1 + 4693850 + (3000+new_source_z1) / 10000 * math.tan(math.radians(10))

    replacement_x = f'latorUTM: {new_source_y1}'
    replacement_y = f'longorUTM: {new_source_x1}'
    replacement_z = f'depth: {new_source_z1}'

    content = re.sub(r'latorUTM:\s+\S+', replacement_x, content)
    content = re.sub(r'longorUTM:\s+\S+', replacement_y, content)
    content = re.sub(r'depth:\s+-\S+', replacement_z, content)

    with open(cmtsolution_path, 'w') as file:
        file.write(content)
    logging.info(f"CMTSOLUTION modified with new source position: ({new_source_x}, {new_source_z})")
    
    return new_source_x1, new_source_y1, new_source_z1

# 提交SPECfem3D作业
def submit_specfem_job():
    command = ["sbatch", specfem_sh_path]
    try:
        result = subprocess.run(command, cwd=working_directory, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            logging.error(f"Failed to submit SPECfem3D job. STDOUT: {result.stdout} STDERR: {result.stderr}")
            return None
        job_id = re.search(r"Submitted batch job (\d+)", result.stdout).group(1)
        logging.info(f"SPECfem3D job submitted with job ID: {job_id}")
        return job_id
    except Exception as e:
        logging.error(f"An error occurred while submitting SPECfem3D job: {e}")
        return None

# 检查SPECfem3D输出结果是否存在并完成
def is_specfem_output_complete(expected_x, expected_y, expected_z):
    try:
        cmtsolution_path = os.path.join(output_files_dir, "CMTSOLUTION")
        cmtsolution_exists = os.path.exists(cmtsolution_path)
        first_station_exists = all(
            os.path.exists(os.path.join(output_files_dir, f"HD.CZ1.{component}.sema")) for component in ['CXX', 'CXY', 'CXZ']
        )
        last_station_exists = all(
            os.path.exists(os.path.join(output_files_dir, f"HD.CZ4096.{component}.sema")) for component in ['CXX', 'CXY', 'CXZ']
        )

        # 如果任何一个文件不存在，返回 False
        if not cmtsolution_exists or not first_station_exists or not last_station_exists:
            logging.info("One or more required files do not exist, job may still be running.")
            return False

        # 检查 CMTSOLUTION 文件中的参数是否一致
        with open(cmtsolution_path, 'r') as file:
            content = file.readlines()
        
        actual_x = actual_y = actual_z = None
        for line in content:
            if line.startswith('latorUTM'):
                actual_y = float(line.split()[1])
            elif line.startswith('longorUTM'):
                actual_x = float(line.split()[1])
            elif line.startswith('depth'):
                actual_z = float(line.split()[1])
        
        parameters_match = (actual_x == expected_x and actual_y == expected_y and actual_z == expected_z)
        
        logging.info(f"CMTSOLUTION exists: {cmtsolution_exists}")
        logging.info(f"First station files exist: {first_station_exists}")
        logging.info(f"Last station files exist: {last_station_exists}")
        logging.info(f"Parameters match: {parameters_match}")
        
        return cmtsolution_exists and first_station_exists and last_station_exists and parameters_match
    except FileNotFoundError:
        logging.info("CMTSOLUTION or station files not found, job may still be running.")
        return False

# 检查文件是否为空
def is_file_empty(file_path):
    try:
        with open(file_path, 'r') as file:
            data = np.loadtxt(file, usecols=[1])
            return data.size == 0
    except Exception as e:
        logging.error(f"An error occurred while checking file {file_path}: {e}")
        return True

# 创建源张量
def create_source_tensor(max_x, max_z, source_x, source_z, radius=3):
    source_tensor = np.zeros((max_x, max_z))
    for x in range(max_x):
        for z in range(max_z):
            dist = np.sqrt((x - source_x) ** 2 + (z - source_z) ** 2)
            if dist <= radius:
                source_tensor[x, z] = max(0, 1 - dist / radius)
            else:
                source_tensor[x, z] = 0
    return source_tensor

# 处理位移数据
def process_displacement_data(output_files_dir):
    displacement_tensor = np.zeros((50, 64, 64, 3))  
    for station in range(1, 4097):
        for component in ['CXX', 'CXY', 'CXZ']:
            file_path = os.path.join(output_files_dir, f"HD.CZ{station}.{component}.sema")
            if not os.path.exists(file_path):
                logging.warning(f"Warning: {file_path} does not exist.")
                continue
            if is_file_empty(file_path):
                logging.warning(f"Warning: {file_path} is empty.")
                continue
            with open(file_path, 'r') as file:
                try:
                    data = np.loadtxt(file, usecols=[1])
                    for step in range(50):
                        index = {'CXX': 0, 'CXY': 1, 'CXZ': 2}[component]
                        displacement_tensor[step, (station-1) // 64, (station-1) % 64, index] = data[step * 100]
                except Exception as e:
                    logging.error(f"An error occurred while processing file {file_path}: {e}")
                    continue
    return displacement_tensor

# 保存数据到 HDF5 文件
def save_to_hdf5(displacement_data, source_tensor, output_hdf5_path, batch_index, simulation_index):
    hdf5_file_path = os.path.join(output_hdf5_path, f'displacement_data_batch{batch_index}_simulation{simulation_index}.h5')
    try:
        with h5py.File(hdf5_file_path, 'w') as hdf:
            hdf.create_dataset('displacement', data=displacement_data)
            hdf.create_dataset('source_tensor', data=source_tensor)
        logging.info(f"Saved HDF5 file: {hdf5_file_path}")
    except Exception as e:
        logging.error(f"An error occurred while saving HDF5 file: {hdf5_file_path}, error: {e}")

# 保存位置信息
def save_positions(file_path, positions):
    try:
        with open(file_path, 'w') as f:
            for pos in positions:
                f.write(f"{pos[0]} {pos[1]} {pos[2]} {pos[3]} {pos[4]}\n")
        logging.info(f"Saved positions to file: {file_path}")
    except Exception as e:
        logging.error(f"An error occurred while saving positions: {e}")

# 读取 CMTSOLUTION 文件
def read_cmtsolution(cmtsolution_path):
    try:
        with open(cmtsolution_path, 'r') as file:
            content = file.readlines()

        latorUTM = longorUTM = depth = None
        for line in content:
            if line.startswith('latorUTM'):
                latorUTM = float(line.split()[1])
            elif line.startswith('longorUTM'):
                longorUTM = float(line.split()[1])
            elif line.startswith('depth'):
                depth = float(line.split()[1])

        return latorUTM, longorUTM, depth
    except Exception as e:
        logging.error(f"An error occurred while reading CMTSOLUTION file: {e}")
        return None, None, None

# 合并HDF5文件
def merge_hdf5_files(batch_index, num_simulations):
    merged_file_path = os.path.join(output_hdf5_path, f'displacement_data{batch_index}.h5')
    with h5py.File(merged_file_path, 'w') as merged_hdf:
        for simulation_index in range(num_simulations):
            hdf5_file_path = os.path.join(output_hdf5_path, f'displacement_data_batch{batch_index}_simulation{simulation_index}.h5')
            with h5py.File(hdf5_file_path, 'r') as hdf:
                displacement_data = hdf['displacement'][:]
                source_tensor = hdf['source_tensor'][:]
                merged_hdf.create_dataset(f'displacement{simulation_index + 1}', data=displacement_data)
                merged_hdf.create_dataset(f'source{simulation_index + 1}', data=source_tensor)
            logging.info(f"Merged file {hdf5_file_path} into {merged_file_path}")

# 主程序
if __name__ == "__main__":
    try:
        source_positions = []
        transformed_positions = []
        num_batches = 5  # 定义批次数量
        num_simulations = 100  # 每个批次的模拟次数

        for batch in range(num_batches):
            for simulation in range(num_simulations):
                new_source_x = np.random.uniform(0, 64)
                new_source_z = np.random.uniform(0, 64)

                new_source_x1, new_source_y1, new_source_z1 = modify_cmtsolution(new_source_x, new_source_z)

                job_id = submit_specfem_job()
                if not job_id:
                    logging.error("Job submission failed, stopping further execution.")
                    break

                # 等待作业完成，每20秒检查一次输出文件
                while True:
                    if is_specfem_output_complete(new_source_x1, new_source_y1, new_source_z1):
                        # 再次检查是否有空文件
                        empty_files_found = False
                        for station in [1, 4096]:
                            for component in ['CXX', 'CXY', 'CXZ']:
                                file_path = os.path.join(output_files_dir, f"HD.CZ{station}.{component}.sema")
                                if is_file_empty(file_path):
                                    logging.warning(f"File {file_path} is empty, continuing to wait...")
                                    empty_files_found = True
                                    break
                            if empty_files_found:
                                break
                        if not empty_files_found:
                            logging.info("SPECfem3D output is complete.")
                            break
                    logging.info(f"Job {job_id} is still running, waiting...")
                    time.sleep(100)  # 以秒为单位等待

                displacement_data = process_displacement_data(output_files_dir)

                latorUTM, longorUTM, depth = read_cmtsolution(cmtsolution_path)

                if latorUTM is None or longorUTM is None or depth is None:
                    logging.error("Failed to read CMTSOLUTION, stopping further execution.")
                    break


                x_coord = 64 * (longorUTM - 720000) / 20000
                y_coord =( -3000 - depth) / 10000*64

                source_tensor = create_source_tensor(64, 64, x_coord, y_coord, radius=3)

                source_positions.append((new_source_x, new_source_z, latorUTM, longorUTM, depth))
                transformed_positions.append((x_coord, y_coord, latorUTM, longorUTM, depth))

                save_to_hdf5(displacement_data, source_tensor, output_hdf5_path, batch, simulation)

            merge_hdf5_files(batch, num_simulations)

        save_positions(source_positions_path, source_positions)
        save_positions(transformed_positions_path, transformed_positions)

        logging.info("Finished all simulations")
    except Exception as e:
        logging.error(f"An error occurred in the main execution: {e}")
