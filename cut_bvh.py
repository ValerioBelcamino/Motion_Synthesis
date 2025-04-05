file = '/home/belca/Downloads/HDM_bk_04-01_02_120_stageii.bvh'
file2 = '/home/belca/Downloads/HDM_bk_04-01_02_120_stageii2.bvh'

with open (file, 'r') as f:
    lines = f.readlines()

print(lines[315])
header = lines [:316]
lines = lines [316:]

multi = 3

# lines = lines [20*multi:]
lines = lines [10*multi:210*multi]

header.extend(lines)
lines = header

lines = ''.join(lines)

with open(file2, 'w+') as f2:
    f2.write(lines)