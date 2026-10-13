import os, wfdb
from collections import Counter
DATABASE_PATH = r"C:\Users\orduz\Desktop\EMG INV\dates2"
records = sorted(f[:-4] for f in os.listdir(DATABASE_PATH) if f.endswith(".hea"))
c = Counter()
for r in records:
    c[wfdb.rdheader(os.path.join(DATABASE_PATH, r)).sig_name[0]] += 1
print(c)