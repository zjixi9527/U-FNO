"""Find where ASCII header ends and binary data begins."""
import os
import struct

p = r"G:\小论文规划\fno三维小论文\计算地球物理\修改\门源1.grd"

with open(p, "rb") as f:
    data = f.read()

print(f"Total file size: {len(data)} bytes")

# Find where the ASCII header likely ends
# Look for the first byte after the 5th CRLF that starts binary data
# Also try to find where non-ASCII data begins

crlf_positions = []
for i in range(len(data) - 1):
    if data[i] == 0x0D and data[i+1] == 0x0A:
        crlf_positions.append(i + 2)  # position after \r\n

print(f"CRLF count: {len(crlf_positions)}")
print(f"First 10 CRLF end positions: {crlf_positions[:10]}")

# The 5th CRLF ends the header
if len(crlf_positions) >= 5:
    header_end_guess = crlf_positions[4]
    print(f"5th CRLF ends at byte: {header_end_guess}")
    remaining = len(data) - header_end_guess
    expected = 301 * 301 * 4
    print(f"Remaining: {remaining}, expected binary: {expected}")

    # Check if bytes after header_end are valid float32
    # Try reading the "extra" bytes
    first_10_floats = struct.unpack('<' + 'f' * 10, data[header_end_guess:header_end_guess + 40])
    print(f"First 10 float32 (little-endian) after header: {first_10_floats}")

    first_10_floats_be = struct.unpack('>' + 'f' * 10, data[header_end_guess:header_end_guess + 40])
    print(f"First 10 float32 (big-endian) after header: {first_10_floats_be}")

# Search for where ASCII text transitions to binary
# Scan line by line and check if each line is printable ASCII
print("\n=== Scanning line by line for ASCII→binary transition ===")
line_start = 0
ascii_line_count = 0
binary_first_seen = -1

for pos in crlf_positions + [len(data)]:
    line = data[line_start:pos]
    is_ascii = all(32 <= b < 127 or b in (0x0D, 0x0A) for b in line)
    if not is_ascii and binary_first_seen < 0:
        binary_first_seen = line_start
        print(f"First non-ASCII line starts at byte: {line_start}")
        print(f"Last ASCII line is at: {crlf_positions[ascii_line_count - 1] if ascii_line_count > 0 else 0}")
        print(f"Total ASCII lines: {ascii_line_count}")
        break
    line_start = pos
    ascii_line_count += 1

if binary_first_seen >= 0:
    # Check if remaining data matches expected size
    remaining_from_binary = len(data) - binary_first_seen
    print(f"Remaining from binary start: {remaining_from_binary}")
    print(f"Expected float32: {301*301*4} = {301*301*4}")
    print(f"Expected float64: {301*301*8} = {301*301*8}")

    # Try reading data from binary_first_seen as float32
    if remaining_from_binary == 301 * 301 * 4:
        print("\n=== MATCH: Data starts at binary_first_seen with float32 ===")
        test_vals = struct.unpack('<' + 'f' * 5, data[binary_first_seen:binary_first_seen + 20])
        print(f"First 5 values (LE): {test_vals}")
    elif remaining_from_binary == 301 * 301 * 8:
        print("\n=== MATCH: Data starts at binary_first_seen with float64 ===")
        test_vals = struct.unpack('<' + 'd' * 5, data[binary_first_seen:binary_first_seen + 40])
        print(f"First 5 values (LE): {test_vals}")

# Also try: maybe the entire file after some specific byte is binary
# Check common header sizes
for header_size in [149, 300, 500, 1000, 2000, 5000, 10000, 20000]:
    if header_size < len(data) - 301*301*4:
        remaining = len(data) - header_size
        diff = remaining - 301*301*4
        if abs(diff) < 100:
            print(f"\nHeader size {header_size} leaves {remaining} bytes (diff from expected: {diff})")
