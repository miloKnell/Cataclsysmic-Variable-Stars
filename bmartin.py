in_file='bmartin_data.txt'
out_file = 'bmartin_data_pretty.csv'


expected_len = 5 #expected number of columns
data = []
with open(in_file,'r') as f:
    for line in f.readlines():
        if line.strip() == '': #check if the line is empty
            continue
        line = line.strip().split('? ') #split it using a sep
        try:
            [float(x.strip()) for x in line] #make sure that all data can be turned into a float
        except:
            print('FLOAT CONVERSION ERROR')
            print(line)
            
        if len(line) != expected_len: #make sure that everything is the right length
            print(line)

        data.append(line)

jd_col = 0
mag_col = 2
jd = [line[jd_col] for line in data]
mag = [line[mag_col] for line in data]

with open(out_file,'w') as f: #write
    f.write('JD, magnitude\n')
    for line in data:
        f.write(', '.join(line)+'\n')
