import pint
import numpy as np
from scipy.integrate import odeint
from pydantic import BaseModel
from pydantic_numpy import NDArray
from pydantic_pint import PydanticPintQuantity as PPQuantity

# from https://github.com/hgrecco/pint/issues/203

ureg = pint.UnitRegistry()
Qty = ureg.Quantity

@ureg.wraps(ret = '=A**2', args = ('=A', '=A'), strict = True)
def sqsum(x, y):
    return x * x  + 2 * x * y + y * y

@ureg.wraps('m', ('m', 'm/s', 's', 's**-2', 's**-1'))
def solve(s0, v0, t, a, b):
    def diff(y, t_):
        val = [ y[1], (a*y[0]+b*y[1]) ]
        return val
    #print(s0, v0, t, a, b)
    y0 = np.asarray([s0, v0])
    return odeint(diff, y0, t, full_output=False)



t = np.linspace(0.0,10.0,101) * ureg.second
a = Qty(-2.0,'s**-2')
b = Qty(-0.1,'s**-1')
s0 = Qty(0.05,'m')
v0 = Qty(0.2,'m/s')
res = solve(s0, v0, t, a, b)
print(res)
