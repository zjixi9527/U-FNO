"""Scan for ASCII→binary transition in GRD file."""
import struct

p = r"G:\小论文规划\fno三维小论文\计算地球物理\修改\门源1.grd"

with open(p, "rb") as f:
    data = f.read()

total = len(data)
print(f"Total size: {total}")

# The file might have many lines of ASCII data (not just 5-line header)
# Let's find where printable ASCII text ends
# Scan byte by byte for the transition

line_starts = [0]
for i in range(1, total - 1):
    if data[i] == 0x0D and data[i+1] == 0x0A:
        line_starts.append(i + 2)

print(f"Total lines: {len(line_starts)}")

# Check each line for ASCII content
binary_line = -1
ascii_count = 0
for idx, ls in enumerate(line_starts):
    line_end = line_starts[idx + 1] if idx + 1 < len(line_starts) else total
    line = data[ls:line_end]

    # A line is "ASCII" if all bytes are printable ASCII
    is_ascii = all(32 <= b < 127 for b in line)

    if not is_ascii and binary_line < 0:
        binary_line = idx
        print(f"\nASCII lines: {ascii_count}")
        print(f"First non-ASCII line index: {idx}, starts at byte {ls}")
        # Show the last ASCII line
        prev_ls = line_starts[idx - 1] if idx > 0 else 0
        print(f"Last ASCII line: {data[prev_ls:ls].decode('ascii', errors='replace')}")
        # Show first bytes of binary line (hex)
        first_20 = line[:20]
        print(f"Binary line first 20 bytes (hex): {' '.join(f'{b:02x}' for b in first_20)}")

        # Check what remains after this binary line
        remaining = total - ls
        print(f"Remaining from this line: {remaining}")
        expected = 301 * 301 * 8  # float64
        print(f"Expected float64: {expected}")

        # Maybe this binary line contains both text and binary?
        # Check where in the line non-ASCII starts
        first_non_ascii = -1
        for bi, b in enumerate(line):
            if not (32 <= b < 127):
                first_non_ascii = bi
                break
        if first_non_ascii >= 0:
            print(f"Non-ASCII starts at byte {first_non_ascii} within line")
            # Bytes after this point should be the binary data
            binary_data_start = ls + first_non_ascii
            remaining_from_binary = total - binary_data_start
            expected_f32 = 301 * 301 * 4
            expected_f64 = 301 * 301 * 8
            print(f"Binary data remaining: {remaining_from_binary}")
            print(f"Expected float32: {expected_f32}, diff: {remaining_from_binary - expected_f32}")
            print(f"Expected float64: {expected_f64}, diff: {remaining_from_binary - expected_f64}")

            # Try reading float64 values from binary_data_start
            if remaining_from_binary >= 40:
                vals64 = struct.unpack('<' + 'd' * 5, data[binary_data_start:binary_data_start + 40])
                print(f"First 5 float64 at binary offset: {vals64}")
                vals32 = struct.unpack('<' + 'f' * 5, data[binary_data_start:binary_data_start + 20])
                print(f"First 5 float32 at binary offset: {vals32}")

        break
    elif is_ascii:
        ascii_count = idx + 1

if binary_line < 0:
    print("All lines appear to be ASCII - no binary data found?")

    # Maybe the file has some lines that look ASCII but aren't
    # Let's check lines after the 5th one for non-printable chars
    for idx in range(5, min(50, len(line_starts))):
        ls = line_starts[idx]
        line_end = line_starts[idx + 1] if idx + 1 < len(line_starts) else total
        line = data[ls:line_end]
        has_non_printable = any(b < 32 or b >= 127 for b in line)
        if has_non_printable:
            print(f"Line {idx} has non-printable chars: {line[:80]}")
