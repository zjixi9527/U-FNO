"""Figure out GRD file layout — is it ASCII grid with per-cell metadata?"""
p = r"G:\小论文规划\fno三维小论文\计算地球物理\修改\门源1.grd"

with open(p, "rb") as f:
    data = f.read()

total = len(data)
print(f"Total size: {total}")

# Split by CRLF
text = data.decode("ascii", errors="replace")
lines = text.split("\r\n")
print(f"Total lines: {len(lines)}")
print(f"\nFirst 10 lines:")
for i, line in enumerate(lines[:10]):
    print(f"  Line {i}: [{line[:150]}]")

# Check line 5 onwards for pattern
print(f"\nLine 5: [{lines[5][:150]}]")
print(f"Line 6: [{lines[6][:150]}]")
print(f"Line 7: [{lines[7][:150]}]")
print(f"Line 8: [{lines[8][:150]}]")

# How many non-empty lines after line 5?
non_empty = [l for l in lines[5:] if l.strip()]
print(f"\nNon-empty lines after line 5: {len(non_empty)}")
print(f"Expected: 301*301 = {301*301}")

# This might be a header + one value per line format (like ESRI ASCII grid)
# But 301*301 = 90601, and we have ~9631 lines after line 5
# That's close to 9631 lines. Let me check...

print(f"\nLines 5 to 5+301*301: that would be lines 5 to {5+301*301}")
if len(lines) >= 5 + 301*301:
    print(f"File has {len(lines)} lines, enough for 301*301 values")
else:
    print(f"File has {len(lines)} lines, NOT enough for 301*301 values")

# Maybe it's 301 rows of 301 values each, space-separated
# Check line 5 for number of values
line5_parts = lines[5].strip().split()
print(f"\nLine 5 has {len(line5_parts)} space-separated values")
print(f"First 5: {line5_parts[:5]}")

# Check if each subsequent line has 301 values
for i in range(5, min(10, len(lines))):
    parts = lines[i].strip().split()
    print(f"Line {i} has {len(parts)} values")

# If there are ~9631 lines after line 5, and we need 301*301 = 90601 values,
# then each line must have multiple values
if len(non_empty) > 0:
    total_values_after_line5 = 0
    for line in non_empty[:20]:
        total_values_after_line5 += len(line.strip().split())
    print(f"\nTotal values in first 20 non-empty lines: {total_values_after_line5}")
    print(f"Average values per line: {total_values_after_line5/20}")
    print(f"Expected values per line (301*301/9631): {90601/9631:.1f}")
