"""Identify GRD file format by examining structure."""
import struct

p = r"G:\小论文规划\fno三维小论文\计算地球物理\修改\门源1.grd"

with open(p, "rb") as f:
    data = f.read()

total = len(data)
print(f"Total: {total} bytes")

# First 4 bytes (the hex shows 44 53 41 41 = "DSAA" but at position 4 in debug output)
# Wait - the debug said "Non-ASCII starts at byte 4 within line"
# and the hex starts with 44 53 41 41 at position 0
# So the file literally starts with "DSAA\r\n"

# Let's verify: read bytes 0-20 as hex
print(f"\nFirst 20 bytes (hex): {' '.join(f'{b:02x}' for b in data[:20])}")
print(f"First 20 bytes (text): {data[:20]}")

# OK so DSAA starts at byte 0. The debug_grd3 script was confused because
# it split on CRLF first, then the "line" starts at 0 with "DSAA\r\n...".
# Actually looking back, line_starts[0] = 0, so the first "line" starts at byte 0
# which is "DSAA\r\n...". The "non-ASCII at byte 4" was checking within line content
# but the line includes the CRLF. Let me just work directly.

# Find all CRLF positions
crlfs = []
i = 0
while i < len(data) - 1:
    if data[i] == 0x0D and data[i+1] == 0x0A:
        crlfs.append(i + 2)
        i += 2
        continue
    i += 1

print(f"\nCRLF count: {len(crlfs)}")
print(f"First 6 CRLF end positions: {crlfs[:6]}")

# Lines: line 0 = bytes 0..crlfs[0], line 1 = crlfs[0]..crlfs[1], etc
lines_raw = []
for i in range(len(crlfs)):
    start = crlfs[i-1] if i > 0 else 0
    end = crlfs[i]
    lines_raw.append(data[start:end])

print(f"\nParsed {len(lines_raw)} lines")
print(f"Line 0 (header row 1): {lines_raw[0][:80]}")
print(f"Line 1 (header row 2): {lines_raw[1][:80]}")
print(f"Line 2 (header row 3): {lines_raw[2][:80]}")
print(f"Line 3 (header row 4): {lines_raw[3][:80]}")
print(f"Line 4 (header row 5): {lines_raw[4][:80]}")

# Now check what happens after line 5
# The remaining data starts at crlfs[5]
binary_start = crlfs[5]
remaining = total - binary_start
print(f"\nAfter 5th CRLF:")
print(f"  binary_start byte: {binary_start}")
print(f"  remaining: {remaining}")
print(f"  expected 301*301*4 = {301*301*4}")
print(f"  expected 301*301*8 = {301*301*8}")

# remaining doesn't match float32 (362404) or float64 (724808)
# diff from float32: {remaining - 301*301*4}
# diff from float64: {remaining - 301*301*8}

# Maybe the grid dimensions from the header are wrong, or the header is longer
# Let's check if lines 5 onwards contain readable text
line5_text = lines_raw[5].decode("ascii", errors="replace")
print(f"\nLine 5 starts with: {line5_text[:200]}")

# Check if this is a text grid (ASCII format with numbers)
try:
    vals = line5_text.strip().split()
    print(f"  Split into {len(vals)} parts")
    # Try parsing first few as floats
    for v in vals[:5]:
        print(f"    {v} -> {float(v)}")
except:
    print("  Not parseable as text")

# If lines 5+ are text numbers, try reading as ASCII grid
# Check if line 6 exists and has similar structure
if len(lines_raw) > 6:
    line6 = lines_raw[6].decode("ascii", errors="replace")
    print(f"\nLine 6 starts with: {line6[:200]}")
    vals6 = line6.strip().split()
    print(f"  Split into {len(vals6)} parts")

# Check a few more lines
for i in range(5, min(12, len(lines_raw))):
    line = lines_raw[i].decode("ascii", errors="replace").strip()
    vals = line.split() if line else []
    print(f"Line {i}: {len(vals)} values, first 3: {vals[:3]}")

# Count total non-empty lines from line 5 onwards
non_empty_from_5 = 0
total_values_from_5 = 0
for i in range(5, len(lines_raw)):
    line = lines_raw[i].decode("ascii", errors="replace").strip()
    if line:
        non_empty_from_5 += 1
        total_values_from_5 += len(line.split())

print(f"\nFrom line 5 onwards:")
print(f"  Non-empty lines: {non_empty_from_5}")
print(f"  Total values: {total_values_from_5}")
print(f"  Expected: 301*301 = {301*301}")
