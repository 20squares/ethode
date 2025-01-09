from simple import *

#############################
# CONTENT BELOW
#############################

# simple ESC model

PB = Units.PB
ETH = Units.ETH
PE = 1 / ETH
EPB = Units.EPB
Dless = Units.dimensionless
Block = Units.Block
Price = Units.PriceETHUSD

def ETHx(n:int) -> tuple[ETH,...]:
    x = (ETH, ) * n
    return tuple[*x]

def EPBx(n:int) -> tuple[EPB,...]:
    x = (EPB, ) * n
    return tuple[*x]

# Simple hard-coded functions

@dataclass
class SimpleParams(Params):
    r: float # staking reinvestment fraction in (0,1)
    f: float # tx fees apy
    y: float # issuance apy
    e: float = 1
    
@dataclass
class SimpleESU(ODESim):
    @staticmethod
    def test(v:tuple[float,...], t:float, tol:float = 1e-12) -> bool:
        E,S,U = v
        troo = all(E > 0) & all(S > 0) & all(U > 0)
        troo &= all(abs(E - S - U) < tol)
        return troo
    @staticmethod
    def func(v:tuple[float,...], t:float, p:Params) -> tuple[float,...]:
        E,S,U = v
        dE = p.y * np.sqrt(p.e / S) * S
        fees = p.f * (U / p.e)
        profit = dE + fees
        dS = p.r * profit
        dU = (1 - p.r) * profit - fees
        return dE, dS, dU
        
# Modular functions

@dataclass
class ComplexESU(SimpleESU):
    @staticmethod
    def func(v:tuple[float,...], t:float, p:Params) -> tuple[float,...]:
        # load variables into a dict
        d = {k:v for k,v in zip(('E','S','U'), v)}
        E,S,U = v
        d['t'] = t
        # compute functions
        r = p.renvst(**d)
        f = p.fees(**d)
        dE = p.yield_curve(**d) * S
        profit = dE + f
        dS = r * profit
        dU = (1 - r) * profit - f
        return dE, dS, dU

@dataclass
class LinFeeParams(Params):
    def yield_curve(self, S: float, **kwargs) -> float:
        return self.y * np.sqrt(self.e / S)
    def fees(self, U: float, **kwargs) -> float:
        return self.f * (U / self.e)
    def renvst(self, **kwargs) -> float:
        return self.r

# build on top of previous models
    
@dataclass
class KineticFeeParams(LinFeeParams):
    k: float = 10E4
    def fees(self, U: float, **kwargs) -> float:
        u = U / self.e
        k = self.k / self.e
        return self.f * u ** 2 /(u + k)

@dataclass
class AndersCurveParams(LinFeeParams):
    k: float = 1
    def yield_curve(self, S: float, **kwargs) -> float:
        return self.y * np.sqrt(self.e / S / (1 + self.k * S))


### Inflation model w/ price

InflData = tuple[float,...]

@dataclass
class InflSim(ODESim):
    @staticmethod
    def func(v:InflData, t:float, p:Params) -> InflData:
        # load variables into a dict
        d = {k:v for k,v in zip(('P','E','S','L','C','B'), v)}
        P,E,S,L,C,B = v
        d['t'] = t
        # compute fees
        tx_fees = p.tot_fees_mev(**d)
        burned_fees = p.burned_fees(tx_fees, **d)
        post_burn_fees = tot_fees - burned_fees
        solo_pfees, lsp_pfees = p.split_post_burn(post_burn_fees, **d)
        # compute derivatives
        dP = (p.dlog_utility(**d) - p.dlog_supply(**d)) * P
        dE = (y := p.yield_curve(**d)) * (S + L)
        dS = y * S + solo_pfees - (K := p.usd_cost(**d) / P)
        dL = (r := p.lsp_renvst(**d)) * (lsp_rev := y * L + lsp_pfees)
        dC = K + (1 - r) * lsp_rev - tx_fees
        dB = burned_fees
        return dP, dE, dS, dL, dC, dB
    
@dataclass
class InflParams(Params):
    def tot_fees_mev(self, **kwargs) -> float: pass
    def burned_fees(self, tot_fees:float, **kwargs) -> float: pass
    def split_post_burn(self, post_burn_fees: float, **kwargs) -> tuple[float,float]: pass
    def dlog_utility(self, **kwargs) -> float: pass
    def dlog_supply(self, **kwargs) -> float: pass
    def yield_curve(self, **kwargs) -> float: pass
    def usd_val_cost(self, **kwargs) -> float: pass
    def lsp_renvst(self, **kwargs) -> float: pass
    def lsp_pfees(self, **kwargs) -> float: pass

# Inflation, supply = E
# uses "on paper inflation" rather than circulating float

# Infl supply = C;
# strict "inflation = expansion of unstaked raw float"

@dataclass
class ESCB(ODESim):
    @staticmethod
    def test(v:tuple[float, float, float, float],
             t:float,
             tol:float = 1e-12) -> bool:
        E,S,C,B = v
        troo = all(E > 0) & all(S > 0) & all(C > 0) & all(B > 0)
        troo &= all(abs(E - S - C - B) < tol)
        return troo
    @staticmethod
    def func(v:tuple[float, float, float, float],
             t:float,
             p:Params) -> tuple[float, float, float, float]:
        # load variables into a dict
        E, S, C, B = v
        d = {k:v for k,v in zip(('E','S','C','B'), v)}
        d['t'] = t
        # compute functions
        r = p.renvst(**d)
        f_tot, f_burned = p.fees(**d)
        f_reward = f_tot - f_burned
        # compute derivatives
        dE = p.issuance(**d) * S
        profit = dE + f_reward
        dS = r * profit
        dC = (1 - r) * profit - f_tot
        dB = f_burned
        return dE, dS, dC, dB
    
@dataclass
class ESCBParams(Params):
    r: float
    f: float
    y: float
    b: float
    e: float = 1
    k: float = 1
    def issuance(self, S: float, **kwargs) -> float:
        return self.y * np.sqrt(self.e / S)
    def renvst(self, **kwargs) -> float:
        return self.r
    def fees(self, C: float, **kwargs) -> tuple[float, float]:
        fees = self.f * C
        return fees, self.burned(fees, **kwargs)
    def burned(self, fees: float, **kwargs) -> float:
        return self.b * fees

