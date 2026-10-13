import sys, numpy, scipy, pandas, sklearn, pywt, wfdb
print(sys.version.split()[0])
for m in (numpy, scipy, pandas, sklearn, pywt, wfdb):
    print(m.__name__, m.__version__)