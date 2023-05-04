#!/nfs/bronfs/uwfs/da00/d59/czhao4/my_python/bin/python


import sys
print(sys.executable)
sys.path.insert(0, "/nfs/bronfs/uwfs/da00/d59/czhao4/my_python/lib/python3.6/site-packages")
import numpy

p=int(sys.argv[1])
q=int(sys.argv[2])
x=float(sys.argv[3])
y=float(sys.argv[4])

print(p, q, x, y)


# dos2unix ~/public_html/hyperbolic-demos/tilings2d.py
