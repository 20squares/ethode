from __future__ import annotations
from typing import TypeVar, Annotated, Generic, Callable, Sequence
from dataclasses import dataclass, make_dataclass

from scipy.integrate import odeint
import numpy as np
import pint

Num = float|int
Quantity = pint.Quantity
Qs = tuple[Quantity, ...]
Nums = tuple[Num, ...]
NumFcn = Callable[Nums, [Num, ...]]
TypedNumFcn = NumFcn

Units = pint.UnitRegistry()
Units.load_definitions('eth_units.txt')
one = 1 * Units.dimensionless
zero = 0 * Units.dimensionless

# fcns

def mag(x: Num|Quantity|Qs) -> Num|Nums:
    if isinstance(x, Quantity):
        return x.magnitude
    elif isinstance(x, Sequence):
        con = type(x)
        return con(mag(e) for e in x)
    else: return x

def wmag(f:Callable|Quantity) -> Callable:
    def wf(*args, _nomag:bool = True, **kwargs):
        out = f(*args, **kwargs)
        return mag(out) if _nomag else out
    return wf

# classes

@dataclass
class Params:
    def _addattr(self, key:str, val:Any):
        setattr(self, key, val)        
    def _parameters(self) -> Iterator[tuple[str, Q, Field]]:
        for k, f in self.__dataclass_fields__.items():
            yield k, getattr(self, k), f
    def _functions(self) -> Iterator[tuple[str, Callable]]:
        for k, v in self.__dict__.items():
            if k[0] != '_' and \
               k not in self.__dataclass_fields__ and \
               callable(v):
                yield k, v
    def _non_dim_args(self) -> dict[str, Num]:
        idata = {}
        fdata = []
        for k,v,f in self._parameters():
            idata[k] = (m := mag(v))
            f.type = Num
            f.default = m
            fdata += [(k, f)]
        Ndp = make_dataclass('NonDimParams',
                             fields = fdata,
                             bases = (self.__class__,))
        ndp = Ndp(**idata)
        for name, fcn in self._functions():
            setattr(ndp, name, wmag(fcn))
        return (ndp, )
        
@dataclass
class Sim:
    ic: tuple[Q, ...]
    tinfo: tuple[Q, Q, Q]
    params: Params
    def sim(self) -> list[tuple[Q, ...]]: pass
    def test(self, tol:float = 1e-12) -> bool: pass
    @staticmethod
    def func(v:tuple[Q,...], t:Q, p:Params) -> bool: pass
    
@dataclass
class ODESim(Sim):
    def sim(self) -> list[tuple[Q, ...]]:
        return odeint(func = wmag(self.func),
                      y0 = mag(self.ic),
                      t = mag(np.arange(*self.tinfo)),
                      args = self.params._non_dim_args())

