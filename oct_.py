import numpy as np

from alternate import ESCBParams, ESCB

y = 1
fs = [2,1,.5,.1,.05,.01,.005,.001,.0005,.0001]
rs = np.arange(0,1.1,.01)

b = .01
data = {}
for f in fs:
    for r in rs:
        print(f'starting {(y,f,r,b)}')
        escb = ESCB(
            ic = (1, .3, .69, .01),
            tinfo = (0, 10**4, 1),
            params = ESCBParams(r,f,y,b))
        data[(r,f,y,b)] = escb.sim()

rf = np.ndarray(dtype=float, shape=(len(rs), len(fs)))
ri = {r:i for i,r in enumerate(rs)}
fi = {f:i for i,f in enumerate(fs)}

for (r, f, y, b), v in data.items():
    e = v[-1]
    rf[ ri[r], fi[f] ] = e[1] / (e[1] + e[2])

plt.contour(rs, np.log(fs), rf.T)
plt.title('S/E =: s(r, ln.f)')
plt.show()
    
