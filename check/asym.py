import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from core import sig, Hb, Policy, lyapunov, lundberg
class AsymP:
    def __init__(self, GL=20., GH=14., rL=0.31, rH=0.33, Delta=0.4, alpha=0.5):
        self.GL, self.GH, self.rL, self.rH, self.alpha = GL, GH, rL, rH, alpha
        self.mL, self.mH = 0.5 - Delta/2, 0.5 + Delta/2
    def pL(self, x): return sig(self.GL*(self.rL-np.abs(x-self.mL)))
    def pH(self, x): return sig(self.GH*(self.rH-np.abs(x-self.mH)))
P = AsymP()
# confounding point: GL(rL-(x-mL)) = GH(rH-(mH-x))
xc = (P.GL*(P.rL+P.mL) - P.GH*(P.rH-P.mH))/(P.GL+P.GH)
pc = P.pL(xc); assert abs(pc-P.pH(xc)) < 1e-12
wc = P.GH/(P.GL+P.GH); lc = np.log(wc/(1-wc))
print("xc", xc, "pc", pc, "wc", wc, "ellc", lc)
# check that the pragmatic optimum at w_c is x_c
x = np.linspace(0,1,200001); pb = wc*P.pL(x)+(1-wc)*P.pH(x); print("argmax pbar at wc", x[np.argmax(pb)], "max-pc", pb.max()-pc)
lam = 1e6
pol = Policy(P, lam, epistemic=False)
# tabulate on signed log grid of delta-ell
mag = 10**np.linspace(-6, np.log10(30), 3000)
xp = pol.choose(sig(lc+mag)); xm = pol.choose(sig(lc-mag))
lg = np.log10(mag)
def x_of(dl):
    a = np.clip(np.abs(dl), mag[0], mag[-1]); k = np.clip(np.searchsorted(lg, np.log10(a))-1, 0, len(lg)-2)
    f = (np.log10(a)-lg[k])/(lg[k+1]-lg[k])
    x0 = np.where(dl>=0, xp[k], xm[k]); x1 = np.where(dl>=0, xp[k+1], xm[k+1])
    jump = np.abs(x1-x0) > 1e-3
    return np.where(jump, np.where(f<0.5, x0, x1), x0+f*(x1-x0))
# linear-response check
s = pc*(1-pc)
K_theory = 1/(2*pc-1)
for d in [1e-4, 1e-3, 1e-2]:
    xx = x_of(np.array([d]))[0]
    up = np.log(P.pL(xx))-np.log(P.pH(xx)); dn = np.log1p(-P.pL(xx))-np.log1p(-P.pH(xx))
    print("dl", d, "U1-1 / theory", (up/d)/(K_theory*(1-pc)), " (U0-1) / theory", (dn/d)/(-K_theory*pc))
m, v, K = lyapunov(np.inf, pc); nu = lundberg(np.inf, pc)
print("symmetric-formula Lambda", m, "nu", nu)
rng = np.random.default_rng(3)
res = []
for d0 in [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]:
    n = 30000
    dl = d0*np.where(rng.random(n)<0.5, 1, -1)
    alive = np.ones(n, bool); learned = np.zeros(n, bool)
    for t in range(1500):
        xx = x_of(dl); pl, ph = P.pL(xx), P.pH(xx)
        a = rng.random(n) < pl       # truth L
        inc = np.where(a, np.log(pl)-np.log(ph), np.log1p(-pl)-np.log1p(-ph))
        dl = np.where(alive, dl+inc, dl)
        learned |= (lc+dl) > 5
        alive &= ~learned & (np.abs(dl) > 1e-5)
    res.append((d0, learned.mean()))
    print(d0, learned.mean(), flush=True)
res = np.array(res)
print("fitted nu", np.polyfit(np.log(res[:,0]), np.log(res[:,1]), 1)[0], "theory", nu)
