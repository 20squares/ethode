import numpy as np
import matplotlib.pyplot as plt

from ex import *

y0 = 1
fs = np.r_[.0001, .0005, .001, .005, .01, .02, .05, .08, .1, .2, .5, .8]
rs = np.arange(0,1,.05)
b0 = .01

data = {}
for f in fs:
    for r in rs:
        print(f'{r}, {f}')
        escb = ESCB(
            ic = (1, .3, .6, .1),
            tinfo = (0, 10**5, 1),
            params = ESCBParams(r = r, f = f, y = y0, b = b0))
        data[(r,f)] = escb.sim()

rf = np.ndarray(dtype=float, shape=(len(rs), len(fs)))
ri = {r:i for i,r in enumerate(rs)}
fi = {f:i for i,f in enumerate(fs)}
for (r,f), d in data.items():
    e = d[-1]
    rf[ ri[r], fi[f] ] = e[1] / (e[1] + e[2])

plt.contour(rs, np.log(fs), rf.T)
plt.title("S/E[r, ln(f/y')]")
plt.show()


data2 = {}
for f in fs:
    for r in rs:
        print(f'{r}, {f}')
        escb = ESCB(
            ic = (1, .3, .6, .1),
            tinfo = (0, 10**5, 1),
            params = AndersCurveParams(r = r, f = f, y = y0, b = b0))
        data2[(r,f)] = escb.sim()
rf2 = np.ndarray(dtype=float, shape=(len(rs), len(fs)))
for (r,f), d in data2.items():
    e = d[-1]
    rf2[ ri[r], fi[f] ] = e[1] / (e[1] + e[2])

plt.contour(rs, np.log(fs), rf2.T)
plt.title("S/E[r, ln(f/y')]")
plt.show()
