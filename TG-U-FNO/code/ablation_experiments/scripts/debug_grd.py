"""Debug script to inspect the GRD file header structure."""
import os
import struct

p = r"G:\小论文规划\fno三维小论文\计算地球物理\修改\门源1.grd"
print("exists:", os.path.exists(p))

if not os.path.exists(p):
    # Try alternate path
    p2 = r"门源1.grd"
    print("Trying alternate path:", os.path.exists(p2))
    p = p2

size = os.path.getsize(p)
print(f"File size: {size} bytes")

with open(p, "rb") as f:
    data = f.read()

total_lines = data.count(b"\n")
print(f"Total newlines: {total_lines}")

# Find all LF positions
lf_positions = [i for i, b in enumerate(data) if b == 0x0A]
print(f"Last LF at byte: {lf_positions[-1] if lf_positions else 'N/A'}")

# Look for the first "DSAA" marker
dsaa_pos = data.find(b"DSAA")
print(f"'DSAA' found at byte: {dsaa_pos}")

# If DSAA is near the start, find the 5th newline after it
if dsaa_pos >= 0:
    # Find newlines after DSAA
    newlines_after_dsaa = [pos for pos in lf_positions if pos > dsaa_pos]
    if len(newlines_after_dsaa) >= 5:
        header_end = newlines_after_dsaa[4] + 1
        header_text = data[dsaa_pos:header_end].decode("ascii", errors="replace")
        print(f"\n=== Header (from DSAA to 5th newline) ===")
        print(header_text)
        print(f"=== End Header ===")
        print(f"header_end from DSAA start: {header_end}")
        remaining = len(data) - header_end
        expected_floats = 301 * 301
        expected_bytes_le = expected_floats * 4
        expected_bytes_be = expected_floats * 4
        print(f"Remaining bytes: {remaining}")
        print(f"Expected binary (301x301x4): {expected_bytes_le}")

        # Check for possible CRLF
        crlf_count = data.count(b"\r\n")
        lf_only_count = total_lines - crlf_count
        print(f"CRLF: {crlf_count}, LF-only: {lf_only_count}")

    else:
        print(f"Only {len(newlines_after_dsaa)} newlines after DSAA")
        # Maybe the file uses a different number of newlines or different header format
        # Print first 30 lines
        lines = data[:min(2000, len(data))].split(b"\n")
        for i, line in enumerate(lines[:20]):
            print(f"Line {i}: {line.decode('ascii', errors='replace')[:200]}")

else:
    # No DSAA marker found - maybe a different format
    print("No 'DSAA' marker found")
    lines = data[:min(5000, len(data))].split(b"\n")
    for i, line in enumerate(lines[:20]):
        print(f"Line {i}: {line.decode('ascii', errors='replace')[:200]}")

# Check if the file might have nx != 301
# Try to find nx, ny pattern in header
if dsaa_pos >= 0:
    newlines_after_dsaa = [pos for pos in lf_positions if pos > dsaa_pos and pos < dsaa_pos + 2000]
    if newlines_after_dsaa:
        header_snippet = data[dsaa_pos:newlines_after_dsaa[-1] + 1].decode("ascii", errors="replace")
        # Try to parse dimensions
        all_lines = header_snippet.strip().split("\n")
        for line in all_lines[:10]:
            parts = line.split()
            # Look for number pairs that could be dimensions
            nums = []
            for p2 in parts:
                try:
                    nums.append(int(p2))
                except ValueError:
                    pass
            if len(nums) >= 2:
                print(f"Possible dimension pair: {nums}")
                expected = nums[0] * nums[1] * 4
                remaining = len(data) - (newlines_after_dsaa[-1] + 1)
                print(f"  Expected binary: {expected}, Actual remaining: {remaining}")
                print(f"  Match: {'YES' if abs(expected - remaining) < expected * 0.01 else 'NO'}")
