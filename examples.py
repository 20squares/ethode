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
USD = Units.USD

def _tuple_units(t:type, n:int, i:tuple = (), f:tuple = ()) -> tuple[type,...]:
    return i + (t, ) * n + f

def ETHx(n:int, i:tuple = (), f:tuple = ()) -> tuple[ETH,...]:
    return _tuple_units(ETH, n, i, f)

def EPBx(n:int, i:tuple = (), f:tuple = ()) -> tuple[ETH,...]:
    return _tuple_units(EPB, n, i, f)

# Simple hard-coded functions

@dataclass
class SimpleParams(Params):
    r: float # staking reinvestment fraction in (0,1)
    f: EPB # tx fees apy
    y: EPB # issuance apy
    e: ETH = 1 * ETH
    
@dataclass
class SimpleESU(ODESim):
    @staticmethod
    def test(v:ETHx(3), t:Block, tol:float = 1e-12) -> bool:
        E,S,U = v
        troo = all(E > 0) & all(S > 0) & all(U > 0)
        troo &= all(abs(E - S - U) < tol)
        return troo
    @staticmethod
    def func(v:ETHx(3), t:Block, p:Params) -> ETHx(3):
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
    def func(v:ETHx(3), t:Block, p:Params) -> ETHx(3):
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
class LinFeeParams(SimpleParams):
    def yield_curve(self, S: ETH, **kwargs) -> ETH:
        return self.y * np.sqrt(self.e / S)
    def fees(self, U: ETH, **kwargs) -> ETH:
        return self.f * (U / self.e)
    def renvst(self, **kwargs) -> EPB:
        return self.r

# build on top of previous models

@dataclass
class KineticFeeParams(LinFeeParams):
    k: ETH = 2 ** 10 * ETH
    def fees(self, U: ETH, **kwargs) -> ETH:
        u = U / self.e
        k = self.k / self.e
        return self.f * u ** 2 /(u + k)

@dataclass
class AndersCurveParams(LinFeeParams):
    k: PE = 1 / ETH
    def yield_curve(self, S: ETH, **kwargs) -> ETH:
        return self.y * np.sqrt(self.e / S / (1 + self.k * S))

### ESCB

@dataclass
class ESCB(ODESim):
    @staticmethod
    def test(v:ETHx(4), t:Block, tol:float = 1e-12) -> bool:
        E,S,C,B = v
        troo = all(E > 0) & all(S > 0) & all(C > 0) & all(B > 0)
        troo &= all(abs(E - S - C - B) < tol)
        return troo
    @staticmethod
    def func(v:ETHx(4), t:Block, p:Params) -> ETHx(4):
        # load variables into a dict
        E, S, C, B = v
        d = {k:x for k,x in zip(('E','S','C','B'), v)}
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
    f: PB
    y: PB
    b: float
    e: ETH = 1
    def issuance(self, S: ETH, **kwargs) -> PB:
        return self.y * np.sqrt(self.e / S)
    def renvst(self, **kwargs) -> float:
        return self.r
    def fees(self, C: ETH, **kwargs) -> tuple[EPB, EPB]:
        print(self.f)
        print(C)
        fees = self.f * C
        return fees, self.burned(fees, **kwargs)
    def burned(self, fees: EPB, **kwargs) -> EPB:
        return self.b * fees

@dataclass
class AndersCurveParams(ESCBParams):
    k: ETH = 1
    def yield_curve(self, S: ETH, **kwargs) -> PB:
        return self.y * np.sqrt(self.e / S / (1 + S / self.k))

    
### Inflation model w/ price

InflData = ETHx(5, (Price, ))

@dataclass
class InflSim(ODESim):
    @staticmethod
    def func(v:InflData, t:Block, p:Params) -> ETHx(4):
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
        dS = y * S + solo_pfees - (k := p.val_cost(**d))
        dL = (r := p.lsp_renvst(**d)) * (lsp_rev := y * L + lsp_pfees)
        dC = K + (1 - r) * lsp_rev - tx_fees
        dB = burned_fees
        return dP, dE, dS, dL, dC, dB

    
@dataclass
class InflParams(Params):
    def supply(self, **kwargs) -> ETH: pass
    def tot_fees_mev(self, **kwargs) -> EPB: pass
    def burned_fees(self, tot_fees:EPB, **kwargs) -> EPB: pass
    def split_post_burn(self, post_burn_fees: EPB, **kwargs) -> EPBx(2): pass
    def dlog_utility(self, **kwargs) -> PB: pass
    def dlog_supply(self, **kwargs) -> PB: pass
    def yield_curve(self, **kwargs) -> PB: pass
    def usd_cost(self, **kwargs) -> USD: pass
    def lsp_renvst(self, **kwargs) -> Dless: pass
    def lsp_pfees(self, **kwargs) -> EPB: pass

# Inflation, supply = E
# uses "on paper inflation" rather than circulating ETH

@dataclass
class EInflParams(InflParams):
    e: ETH = 1 * ETH
    fees_per_eth: PB = .1 * PB
    burn_fraction: float = .01
    mev_advantage: float = .5
    anders_constant: ETH = 2**25 * ETH
    def supply(self, E:ETH, **kwargs) -> ETH:
        return E
    def tot_fees_mev(self, **kwargs) -> EPB:
        return self.fees_per_eth * self.C
    def burned_fees(self, tot_fees:EPB, **kwargs) -> EPB:
        return self.burn_fraction * tot_fees
    def split_post_burn(self, post_burn_fees: EPB, **kwargs) -> EPBx(2):
        m = self.mev_advantage
        return tuple(np.r_[m, 1 - m] * post_burn_fees)
    def yield_curve(self, S:ETH, L:ETH, **kwargs) -> PB:
        staked = S + L
        return self.y * np.sqrt(self.e / staked)
    def val_cost(self, **kwargs) -> USD:
        return self.fixed_cost
    def lsp_renvst(self, **kwargs) -> Dless: pass
    def lsp_pfees(self, **kwargs) -> EPB: pass
    def dlog_utility(self, **kwargs) -> PB: pass        
    def dlog_supply(self, **kwargs) -> PB: pass

