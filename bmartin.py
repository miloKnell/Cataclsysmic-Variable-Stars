import numpy as np
import re

in_file='bmartin_data.txt'
out_file = 'bmartin_data_pretty.csv'


expected_len = 5 #expected number of columns
data = []
with open(in_file,'r') as f:
    for line in f.readlines():
        if line.strip() == '': #check if the line is empty
            continue
        line = line.strip().split('? ') #split it using a sep
        line = [float(x.strip()) for x in line] #make sure that all data can be turned into a float
        if len(line) != expected_len: #make sure that everything is the right length
            print(line)

        data.append(line)


base_jd = 0
norm_factor = 0

jd_col = 0
mag_col = 3


jd = np.array([line[jd_col] for line in data])
mag = np.array([line[mag_col] for line in data])

mag += norm_factor
jd += base_jd

with open(out_file,'w') as f: #write
    f.write('JD, magnitude\n')
    for jd_,mag_ in zip(jd,mag):
        f.write('{}, {}\n'.format(jd_,mag_))
