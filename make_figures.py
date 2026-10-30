"""Figures for the Letter, drawn to the APS figure guidelines for Physical Review Letters:
one column wide (8.6 cm), lettering at least 2 mm high on the printed page and sub- and
superscripts at least 1.5 mm (9 pt STIX, scripts 6.3 pt), lines at least 0.5 pt, panel labels (a), (b), ..., explanations in the captions
rather than on the figures, vector PDF with Times-like STIX fonts. Every series combines
colour with a marker or line style, so it stays distinguishable in greyscale. No green.

usage: python make_figures.py [outdir] [1 2 3 4]
"""
import numpy as np, sys, os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap, LogNorm, PowerNorm, TwoSlopeNorm, ListedColormap
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from scipy.optimize import brentq
from core import Params, Hb, sig, thresholds, lyapunov
from brain import draw_brain

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'figures')
D = os.path.join(HERE, 'data')
W = 3.39          # one PRL column, 8.6 cm
FS = 9.0          # smallest lettering: STIX numerals at 9 pt are 2.1 mm high, their scripts (6.3 pt) 1.5 mm
plt.rcParams.update({
    'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 9,
    'axes.labelsize': 9, 'axes.titlesize': 9, 'xtick.labelsize': FS, 'ytick.labelsize': FS, 'legend.fontsize': FS,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.minor.width': 0.5,
    'ytick.minor.width': 0.5, 'xtick.major.size': 3.0, 'ytick.major.size': 3.0, 'xtick.minor.size': 1.8,
    'ytick.minor.size': 1.8, 'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
    'xtick.major.pad': 2.5, 'ytick.major.pad': 2.5, 'axes.labelpad': 2.0,
    'lines.linewidth': 1.3, 'lines.markersize': 4.5, 'lines.markeredgewidth': 0.9,
    'legend.frameon': False, 'legend.handlelength': 1.6, 'legend.handletextpad': 0.4, 'legend.labelspacing': 0.25,
    'legend.borderaxespad': 0.3, 'legend.columnspacing': 0.9, 'legend.borderpad': 0.2,
    'pdf.fonttype': 42, 'savefig.dpi': 600,
})
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#898781', '#d6d5ce'
BLUE, ORANGE, YELLOW, MAGENTA = '#2a78d6', '#eb6834', '#eda100', '#e87ba4'
LBLUE, NAVY = '#5aaef0', '#0d366b'      # no green anywhere: light blue and navy take its place
DBLUE = '#184f95'
BLUE_RAMP = ['#fcfcfb', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
ORANGE_RAMP = ['#fcfcfb', '#fde4d8', '#f9c2a7', '#f39a73', '#eb6834', '#c44f1e', '#8f3712', '#5c2209']
cm_blue = LinearSegmentedColormap.from_list('blue', BLUE_RAMP)
cm_orange = LinearSegmentedColormap.from_list('orange', ORANGE_RAMP)
P = Params()
TH = thresholds(P)
PC = P.sigma_mid
LAM_A = brentq(lambda l: lyapunov(l, PC)[0], 2, 1e4)


def panel(ax, s, x=0.035, y=0.955, va_='top', **kw):
    """Panel label (a), (b), ... in a free corner of the panel, as in PRL."""
    ax.text(x, y, s, transform=ax.transAxes, ha='left', va=va_, fontsize=9.5, zorder=20, **kw)


def axes_in(fig, l, b, w, h):
    """Axes placed in inches from the lower-left corner of the figure."""
    fw, fh = fig.get_size_inches()
    return fig.add_axes([l / fw, b / fh, w / fw, h / fh])


# --------------------------------------------------------------------------- Fig. 1
def fig1():
    H = 3.40
    fig = plt.figure(figsize=(W, H))
    # (a) factor graph: buyer, host and climate side by side; connectors horizontal or vertical only
    axA = axes_in(fig, 0.0, H - 1.765, W, 1.749)          # 19 x 9.8 units, 0.1784 in per unit on both axes
    axA.set_xlim(0, 19); axA.set_ylim(0, 9.8); axA.axis('off')

    def region(x0, x1, fc, ec):
        axA.add_patch(FancyBboxPatch((x0, 0.1), x1 - x0, 9.6, boxstyle='round,pad=0,rounding_size=0.45',
                                     fc=fc, ec=ec, lw=0.7, zorder=0))
    region(0.10, 9.30, '#eaf2fc', '#9ec5f4')         # buyer
    region(9.55, 13.75, '#fdeee7', '#f9c2a7')        # host
    region(14.00, 18.90, '#f3f3f1', '#c9c8c2')       # climate
    yt = 9.08
    axA.text(0.40, yt, '(a)', ha='left', va='center', fontsize=9.5)
    axA.text(2.05, yt, 'Buyer', ha='left', va='center', fontsize=9, fontweight='bold', color=DBLUE)
    axA.text(9.85, yt, 'Host', ha='left', va='center', fontsize=9, fontweight='bold', color='#8f3712')
    axA.text(14.30, yt, 'Climate', ha='left', va='center', fontsize=9, fontweight='bold', color=INK2)
    yu, yl, xc = 6.95, 2.40, 16.45           # upper and lower loop rows, climate column
    Pg, Px, Pf, Pth = (1.80, yu), (7.20, yu), (10.90, yu), (12.80, yu)
    Pl, Pa, Pb = (1.80, 4.70), (10.90, yl), (7.20, yl)
    Pd, PB, Ptip = (xc, yl), (xc, 4.70), (xc, yu)
    R, S = 0.75, 0.45                        # node radius, factor half-width
    loop = dict(color=BLUE, lw=1.8, solid_capstyle='butt', zorder=2)
    edge = dict(color=INK2, lw=0.9, zorder=2)

    def seg(p, q, **kw):
        axA.plot([p[0], q[0]], [p[1], q[1]], **kw)
    # epistemic loop: belief -> terms -> response -> belief
    for p, q in [(Pl, Pg), (Pg, Px), (Px, Pf), (Pf, Pa), (Pa, Pb), (Pb, (Pl[0], yl)), ((Pl[0], yl), Pl)]:
        seg(p, q, **loop)
    for p, q in [(Px, Pb), (Pf, Pth), (Pa, Pd), (Pd, PB), (PB, Ptip)]:
        seg(p, q, **edge)
    ah = dict(arrowstyle='-|>', color=BLUE, lw=0, mutation_scale=9, shrinkA=0, shrinkB=0, zorder=3)
    for (p, q) in [((4.25, yu), (4.55, yu)), ((8.70, yu), (9.00, yu)), ((Pf[0], 4.95), (Pf[0], 4.65)),
                   ((9.00, yl), (8.70, yl)), ((4.65, yl), (4.35, yl)), ((Pl[0], 3.10), (Pl[0], 3.40)),
                   ((Pl[0], 5.85), (Pl[0], 6.15))]:
        axA.annotate('', xy=q, xytext=p, arrowprops=ah)
    ah2 = dict(arrowstyle='-|>', color=INK2, lw=0, mutation_scale=8, shrinkA=0, shrinkB=0, zorder=3)
    for (p, q) in [((14.95, yl), (15.25, yl)), ((xc, 3.40), (xc, 3.70)), ((xc, 5.70), (xc, 6.00))]:
        axA.annotate('', xy=q, xytext=p, arrowprops=ah2)

    def var(p, lab, fc='white'):
        axA.add_patch(plt.Circle(p, R, fc=fc, ec=INK, lw=0.9, zorder=4))
        axA.text(p[0], p[1], lab, ha='center', va='center', fontsize=9.5, zorder=5)

    def fac(p, fc=INK):
        axA.add_patch(plt.Rectangle((p[0] - S, p[1] - S), 2 * S, 2 * S, fc=fc, ec=INK, lw=0.6, zorder=4))
    var(Pl, r'$\ell_t$'); var(Px, r'$x_t$'); var(Pth, r'$\theta$'); var(Pa, r'$a_t$', fc='#d9d8d2'); var(PB, r'$B_t$')
    fac(Pg); fac(Pf, fc='#c44f1e'); fac(Pb); fac(Pd, fc=INK2); fac(Ptip, fc=INK2)
    axA.text(0.40, yu + 0.55, r'$\exp[-\gamma G_w(x)]$', ha='left', va='bottom', fontsize=FS)
    axA.text(Pf[0], yu + 0.55, r'$p_\theta(x)$', ha='center', va='bottom', fontsize=FS)
    axA.text(Pb[0] + S, yl - 0.62, 'Bayes, Eq. (2)', ha='right', va='top', fontsize=FS)
    axA.text(xc, yl - 0.62, r'$d_t=1-\alpha a_t$', ha='center', va='top', fontsize=FS)
    axA.text(xc, yu + 0.55, r'tips if $B_t<0$', ha='center', va='bottom', fontsize=FS)
    # the buyer's brain inside its epistemic loop; the factors of the loop gain on the loop
    draw_brain(axA, 5.30, 4.62, 2.90, 2.30, light='#f7fbff', lw=0.75, fold_lw=0.5, zorder=3)
    axA.text(2.10, 6.00, r'$d\hat x/d\ell$', ha='left', va='center', fontsize=FS, color=BLUE)
    axA.text(Pf[0] + 0.30, 4.85, r'$a_t-p_{\rm c}$', ha='left', va='center', fontsize=FS, color=BLUE)
    axA.text(2.55, 3.25, r'$K$', ha='left', va='center', fontsize=10, color=DBLUE)

    # (b) responses and information, (c) expected free energy under ignorance
    yb0, hb = 0.40, 1.07
    axB1 = axes_in(fig, 0.46, yb0 + 0.43, 1.12, hb - 0.43)
    axB2 = axes_in(fig, 0.46, yb0, 1.12, 0.38)
    axC = axes_in(fig, 2.04, yb0, 0.86, hb)
    cax = axes_in(fig, 2.95, yb0, 0.06, hb)
    x = np.linspace(0, 1, 2001)
    pp, pm = P.pH(x), P.pL(x)      # p_+ peaks at m_+ = 0.7, p_- at m_- = 0.3
    pb = 0.5 * (pp + pm)
    I = Hb(pb) - 0.5 * Hb(pp) - 0.5 * Hb(pm)
    for ax in (axB1, axB2):
        for m in (P.mL, P.mH):
            ax.axvline(m, color=GRID, lw=0.5, zorder=0)
    axB1.plot(x, pp, color=BLUE, ls='-', marker='o', markevery=250, mfc='white')
    axB1.plot(x, pm, color=ORANGE, ls='--', marker='s', markevery=(125, 250), mfc='white')
    axB1.plot(x, pb, color=INK2, ls=':', lw=1.2)
    axB1.plot([0.5], [PC], marker='D', ms=4.5, color=INK, zorder=5)
    axB1.text(P.mL, 1.06, r'$p_-$', ha='center', va='bottom', fontsize=FS, color=ORANGE)
    axB1.text(P.mH, 1.06, r'$p_+$', ha='center', va='bottom', fontsize=FS, color=BLUE)
    axB1.set_ylim(0, 1.36); axB1.set_yticks([0, 0.5, 1]); axB1.set_yticklabels(['0', '0.5', '1'])
    axB1.set_ylabel('acceptance', labelpad=1)
    plt.setp(axB1.get_xticklabels(), visible=False)
    axB2.fill_between(x, 0, I, color='#cde2fb', lw=0)
    axB2.plot(x, I, color=BLUE, lw=1.1)
    axB2.plot(x, PC - pb, color=INK, ls='-.', lw=1.1)
    axB2.text(0.5, 0.36, r'$I_{1/2}$', ha='center', va='bottom', fontsize=FS, color=BLUE)
    axB2.set_ylim(0, 0.95); axB2.set_yticks([0, 0.5]); axB2.set_yticklabels(['0', '0.5'])
    axB2.set_xlim(0, 1); axB1.set_xlim(0, 1)
    axB2.set_xticks([0, P.mL, 0.5, P.mH, 1]); axB2.set_xticklabels(['0', r'$m_-$', '', r'$m_+$', '1'])
    axB2.set_xlabel(r'terms $x$', labelpad=1)
    panel(axB1, '(b)', x=-0.40, y=1.0, va_='bottom')
    lam = np.logspace(np.log10(0.3), np.log10(10), 300)
    xx = np.linspace(0, 1, 801)
    pp2, pm2 = P.pH(xx), P.pL(xx)
    pb2 = 0.5 * (pp2 + pm2)
    I2 = Hb(pb2) - 0.5 * Hb(pp2) - 0.5 * Hb(pm2)
    G = -lam[:, None] * pb2[None, :] - I2[None, :]
    Grel = G - G.min(axis=1, keepdims=True)
    im = axC.pcolormesh(xx, lam, Grel, cmap=cm_orange, norm=PowerNorm(0.45, vmin=0, vmax=1.5), shading='auto', rasterized=True)
    for l in np.logspace(np.log10(0.3), np.log10(10), 13):
        g = -l * pb2 - I2
        j = np.argmin(g)
        xs = [xx[j], 1 - xx[j]] if abs(xx[j] - 0.5) > 1e-3 else [0.5]
        axC.plot(xs, [l] * len(xs), 'o', ms=4.0, mfc='white', mec=INK, mew=0.8)
    axC.axhline(TH['lam_loc'], color=INK, ls='--', lw=0.9)
    axC.axhline(TH['rho_max'], color=INK, ls='-', lw=0.9)
    axC.text(0.97, TH['rho_max'] * 1.07, r'$\lambda_{\rm c}$', ha='right', va='bottom', fontsize=FS)
    axC.text(0.97, TH['lam_loc'] / 1.07, r'$\lambda_{\rm s}$', ha='right', va='top', fontsize=FS)
    axC.set_yscale('log'); axC.set_ylim(0.3, 10); axC.set_xlim(0, 1)
    axC.set_yticks([1, 10]); axC.set_yticklabels(['1', '10'])
    axC.yaxis.set_minor_formatter(mticker.NullFormatter())
    axC.set_xlabel(r'terms $x$', labelpad=1); axC.set_ylabel(r'precision $\lambda$', labelpad=1)
    axC.set_xticks([0, 0.5, 1]); axC.set_xticklabels(['0', '0.5', '1'])
    cb = fig.colorbar(im, cax=cax, ticks=[0, 0.5, 1, 1.5])
    cb.ax.set_yticklabels(['0', '0.5', '1', '1.5']); cb.ax.tick_params(labelsize=FS, width=0.6, length=2.0)
    cb.outline.set_linewidth(0.6)
    cb.set_label(r'$G_{1/2}-\min G_{1/2}$', fontsize=9, labelpad=2)
    panel(axC, '(c)', x=-0.40, y=1.0, va_='bottom')
    fig.savefig(os.path.join(OUT, 'fig1.pdf'))
    plt.close(fig)


# --------------------------------------------------------------------------- Fig. 2
def fig2():
    from universal import Lam, nu_of, K_a
    fan = np.load(os.path.join(D, 'ergodic_fan.npz'))
    H = 2.98
    fig = plt.figure(figsize=(W, H))
    axF = axes_in(fig, 0.50, 1.93, 2.77, 0.97)
    axA = axes_in(fig, 0.50, 0.40, 2.30, 1.15)
    caxA = axes_in(fig, 2.88, 0.40, 0.07, 1.15)
    # (a) ergodicity breaking: information of standard and ergodic buyers
    t = np.arange(int(fan['T']) + 1); q = fan['aif_q'].copy()
    floor = 1e-16
    # quantiles of the discrete response sequence oscillate from round to round; show their running
    # geometric mean over eleven rounds (the first rounds keep the raw values)
    lq = np.log(np.maximum(q, 1e-300)); sm = lq.copy()
    for k in range(5, len(t)):
        sm[k] = lq[k - 5:min(len(t), k + 6)].mean(axis=0)
    q = np.exp(sm)
    axF.fill_between(t, np.maximum(q[:, 0], floor), q[:, 4], color='#cde2fb', lw=0, zorder=0)
    axF.fill_between(t, np.maximum(q[:, 1], floor), q[:, 3], color='#9ec5f4', lw=0, zorder=1)
    axF.plot(t, q[:, 2], color=BLUE, lw=1.2, marker='o', markevery=15, mfc='white', zorder=4)
    axF.plot(t, fan['aif_mean'], color=INK, lw=1.3, zorder=5)
    axF.plot(t, fan['aif_cumI'], color=ORANGE, lw=1.3, ls=(0, (2.5, 1.5)), zorder=6)
    axF.plot(t, fan['erg_q'][:, 2], color=MAGENTA, lw=1.2, marker='s', markevery=(5, 15), mfc='white', zorder=4)
    Kf = float(lyapunov(20.0, PC)[2]); I0 = float(q[0, 2])
    rate_ens = np.log(1 + Kf ** 2 * PC * (1 - PC)); shift = 100.0     # ensemble guide drawn above the mean it matches
    te = np.linspace(0, 40, 50); tt = np.linspace(0, 150, 50)
    axF.plot(te, shift * I0 * np.exp(rate_ens * te), color=INK2, lw=0.9, ls=':', zorder=3)
    axF.plot(tt, I0 * np.exp(2 * Lam(Kf, PC) * tt), color=INK2, lw=0.8, ls=':', zorder=3)
    axF.set_yscale('log'); axF.set_xlim(0, 150); axF.set_ylim(1e-14, 4e2)
    axF.set_yticks([1e-12, 1e-8, 1e-4, 1]); axF.yaxis.set_minor_locator(mticker.NullLocator())
    axF.set_xticks([0, 50, 100, 150])
    axF.set_xlabel(r'round $t$', labelpad=1); axF.set_ylabel(r'information $\mathcal{I}_t$', labelpad=1)
    axF.text(144, 1.5e-3, r'mean $=\mathcal{I}_0+\sum I$', fontsize=FS, ha='right', va='top')
    axF.text(75, 3e-14, 'median', fontsize=FS, ha='left', va='bottom', color=BLUE)
    axF.text(40, 0.85, 'ergodic', fontsize=FS, ha='left', va='bottom', color=MAGENTA)
    p0, p1 = axF.transData.transform([(10, shift * I0 * np.exp(rate_ens * 10)), (40, shift * I0 * np.exp(rate_ens * 40))])
    ang = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
    axF.text(6, shift * I0 * np.exp(rate_ens * 6) * 2.5, 'ensemble', fontsize=FS, ha='left', va='bottom', color=INK2,
             rotation=ang, rotation_mode='anchor')
    axF.text(62, 4e-9, 'time average', fontsize=FS, ha='left', va='bottom', color=INK2)
    panel(axF, '(a)', y=0.05, va_='bottom')
    # (b) universal phase diagram: Lyapunov exponent (colour) and learning exponent (contours)
    Klo, Khi = 0.40, 8.0
    pg = np.linspace(0.52, 0.99, 260); Kg = np.geomspace(Klo, Khi, 260)
    PP, KK = np.meshgrid(pg, Kg)
    with np.errstate(divide='ignore'):
        LL = np.clip(Lam(KK, PP), -0.6, 2.0)
    cneg = LinearSegmentedColormap.from_list('neg', ['#0d366b', '#256abf', '#6da7ec', '#cde2fb', '#fcfcfb'])
    cpos = LinearSegmentedColormap.from_list('pos', ['#fcfcfb', '#fde4d8', '#f39a73', '#c44f1e', '#5c2209'])
    cmap = ListedColormap(np.vstack([cneg(np.linspace(0, 1, 128)), cpos(np.linspace(0, 1, 128))]))
    imA = axA.pcolormesh(PP, KK, LL, cmap=cmap, norm=TwoSlopeNorm(vmin=-0.3, vcenter=0.0, vmax=0.9), shading='auto', rasterized=True)
    pn = np.linspace(0.53, 0.985, 70); Kn = np.geomspace(1.0, 3.2, 120)
    NU = np.full((len(Kn), len(pn)), np.nan)
    for j, p in enumerate(pn):
        ka = K_a(p)
        for i, k in enumerate(Kn):
            if 1 / p < k < ka:
                NU[i, j] = nu_of(k, p)
    axA.contour(pn, Kn, NU, levels=[0.2, 0.4, 0.6, 0.8], colors='white', linewidths=0.6, linestyles=':')
    axA.plot(pg, [K_a(p) for p in pg], color=INK, lw=1.2)
    axA.plot(pg, 1 / pg, color=INK, ls='--', lw=1.1)
    axA.plot(pg, 1 / (2 * pg - 1), color=INK, ls='-.', lw=1.1)
    pa = brentq(lambda p: K_a(p) - 1 / (2 * p - 1), 0.6, 0.95)
    axA.plot([pa], [K_a(pa)], 'o', ms=4.5, color='white', mec=INK, mew=0.9, zorder=5)
    arr = dict(arrowstyle='-|>', lw=1.0, mutation_scale=8, shrinkA=0, shrinkB=0)
    axA.annotate('', xy=(0.9, 1.27), xytext=(0.9, 6.0), arrowprops=dict(color=INK, **arr))
    axA.annotate('', xy=(0.9, 0.47), xytext=(0.9, 0.84), arrowprops=dict(color=INK, **arr))
    axA.set_yscale('log'); axA.set_xlim(0.52, 0.99); axA.set_ylim(Klo, Khi)
    axA.set_yticks([0.5, 1, 2, 5]); axA.set_yticklabels(['0.5', '1', '2', '5'])
    axA.yaxis.set_minor_formatter(mticker.NullFormatter())
    axA.set_xticks([0.6, 0.7, 0.8, 0.9])
    axA.set_xlabel(r'acceptance probability $p_{\rm c}$', labelpad=1)
    axA.set_ylabel(r'epistemic loop gain $K$', labelpad=1)
    axA.text(0.765, 4.3, r'learning, $\Lambda>0$', fontsize=FS, ha='center', va='center')
    axA.text(0.70, 0.60, r'attractor, $\nu=1$', fontsize=FS, ha='center', va='center')
    axA.text(0.535, 2.62, r'$K_{\rm a}$', fontsize=FS, ha='left', va='bottom')
    axA.text(0.535, 1.70, r'$1/p_{\rm c}$', fontsize=FS, ha='left', va='top', color='white')
    cbA = fig.colorbar(imA, cax=caxA, ticks=[-0.3, 0, 0.3, 0.6, 0.9])
    cbA.ax.set_yticklabels(['$-0.3$', '0', '0.3', '0.6', '0.9'])
    cbA.ax.tick_params(labelsize=FS, width=0.6, length=2.0); cbA.outline.set_linewidth(0.6)
    cbA.set_label(r'Lyapunov exponent $\Lambda$', fontsize=9, labelpad=2)
    panel(axA, '(b)', y=0.05, va_='bottom')
    fig.savefig(os.path.join(OUT, 'fig2.pdf'))
    plt.close(fig)


# --------------------------------------------------------------------------- Fig. 3
def fig3():
    from universal import Lam, nu_of
    ul = json.load(open(os.path.join(D, 'universal_lam.json')))
    un = json.load(open(os.path.join(D, 'universal_nu.json')))
    H = 2.00
    fig = plt.figure(figsize=(W, H))
    axB = axes_in(fig, 0.44, 0.40, 1.22, 1.13)
    axC = axes_in(fig, 2.07, 0.40, 1.22, 1.13)
    mk = {'logit20': ('o', BLUE), 'logit8': ('s', BLUE), 'logit60': ('p', BLUE), 'probit': ('^', LBLUE),
          'gauss': ('v', YELLOW), 'seller': ('D', MAGENTA)}
    names = {'logit20': r'logit, $\Gamma=20$', 'logit8': r'logit, $\Gamma=8$', 'logit60': r'logit, $\Gamma=60$',
             'probit': 'probit', 'gauss': 'Gaussian', 'seller': 'allowance sale'}
    # (a) Lyapunov exponents
    Kg = np.geomspace(0.3, 12, 900)
    for p, lsty in [(0.8, ':'), (0.9, '-'), (0.95, '--')]:
        axB.plot(Kg, Lam(Kg, p), color=INK, ls=lsty, lw=1.0, zorder=1)
    for r in ul:
        key = r['model'].split('_')[0]
        key = 'probit' if key.startswith('probit') else 'gauss' if key.startswith('gauss') else 'seller' if key.startswith('seller') else key
        m, c = mk[key]
        fs = {0.8: 'left', 0.9: 'none', 0.95: 'full'}[round(r['pc'], 2)]     # open 0.90, filled 0.95, half 0.80
        axB.plot(r['K'], r['Lam_sim'], m, color=c, fillstyle=fs if fs != 'none' else 'full',
                 mfc=c if fs != 'none' else 'white', mfcalt='white', zorder=3)
    axB.axhspan(-0.3, 0, color='#e6effa', lw=0, zorder=0)
    axB.axhline(0, color=MUTED, lw=0.5, zorder=0)
    axB.set_xscale('log'); axB.set_xlim(0.42, 12); axB.set_ylim(-0.3, 1.0)
    axB.set_xticks([0.5, 1, 2, 5, 10]); axB.set_xticklabels(['0.5', '1', '2', '5', '10'])
    axB.xaxis.set_minor_formatter(mticker.NullFormatter())
    axB.set_yticks([0, 0.5, 1]); axB.set_yticklabels(['0', '0.5', '1'])
    axB.set_xlabel(r'loop gain $K$', labelpad=1); axB.set_ylabel(r'Lyapunov exponent $\Lambda$', labelpad=1)
    panel(axB, '(a)')
    # (b) learning exponents
    sty2 = {'logit20': 'logit20', 'logit8': 'logit8', 'logit60': 'logit60', 'logit95': 'logit20',
            'probit95': 'probit', 'seller2': 'seller'}
    Kc = np.linspace(0.4, 1.6, 700)
    for p, lsty in [(0.9, '-'), (0.95, '--')]:
        axC.plot(Kc, [nu_of(k, p) for k in Kc], color=INK, ls=lsty, lw=1.0, zorder=1)
    for r in un:
        m, c = mk[sty2[r['model']]]
        filled = r['pc'] > 0.92
        axC.errorbar(r['K'], r['nu_sim'], yerr=r['se'], fmt=m, color=c, mfc=c if filled else 'white',
                     elinewidth=0.6, capsize=0, zorder=3)
    axC.set_xlim(0.42, 1.6); axC.set_ylim(0, 1.18)
    axC.set_xticks([0.5, 1.0, 1.5]); axC.set_xticklabels(['0.5', '1.0', '1.5'])
    axC.set_yticks([0, 0.5, 1]); axC.set_yticklabels(['0', '0.5', '1'])
    axC.set_xlabel(r'loop gain $K$', labelpad=1); axC.set_ylabel(r'learning exponent $\nu$', labelpad=1)
    panel(axC, '(b)', y=0.05, va_='bottom')
    hs = [Line2D([], [], ls='none', marker=mk[k][0], color=mk[k][1], mfc='white') for k in names]
    fig.legend(hs, list(names.values()), loc='upper center', bbox_to_anchor=(0.53, 1.0), ncol=3,
               handlelength=1.0, columnspacing=0.8, handletextpad=0.3)
    fig.savefig(os.path.join(OUT, 'fig3.pdf'))
    plt.close(fig)


# --------------------------------------------------------------------------- Fig. 4
def fig4():
    s = np.load(os.path.join(D, 'surv_curves.npz'))
    es = np.load(os.path.join(D, 'ergodic_surv_b.npz'))
    el = np.load(os.path.join(D, 'ergodic_surv_lam.npz'))
    cal = np.load(os.path.join(D, 'calib_depth.npz'))
    ce = np.load(os.path.join(D, 'calib_ergodic.npz'))
    sn = np.load(os.path.join(D, 'surv_N.npz'))
    sl = np.load(os.path.join(D, 'sale.npz'))
    lo, hi = 1 - P.alpha * P.p_max, 1 - P.alpha * PC
    H = 3.20
    fig = plt.figure(figsize=(W, H))
    axA = axes_in(fig, 0.44, 1.83, 1.22, 0.98)
    axB = axes_in(fig, 2.07, 1.83, 1.22, 0.98)
    axK = axes_in(fig, 0.44, 0.40, 1.22, 0.98)
    axC = axes_in(fig, 2.07, 0.40, 0.86, 0.98)
    caxD = axes_in(fig, 2.98, 0.40, 0.06, 0.98)
    sty = [('aif1.4', BLUE, 'o', '-', r'$\lambda=1.4$'), ('aif3', YELLOW, 'v', '--', r'$\lambda=3$'),
           ('aif20', LBLUE, '^', '-.', r'$\lambda=20$'), ('prag', NAVY, 'D', '-', 'pragmatic')]
    # (a) survival versus allowance
    bb = s['bb']
    axA.axvspan(lo, hi, color='#f3f3f1', lw=0, zorder=0)
    axA.plot(bb, s['oracle'], color=MUTED, ls=':', lw=1.3)
    axA.plot(bb, s['bo'], color=INK, ls=(0, (3, 2)), lw=1.2, zorder=6)
    for key, c, m, ls, lab in sty:
        axA.plot(bb, s[key], color=c, ls=ls, marker=m, markevery=3, mfc='white' if m != 'D' else c, ms=4.0)
    axA.plot(es['bb'], es['erg20'], color=MAGENTA, ls='-', marker='s', markevery=(1, 3), mfc='white', ms=4.0, zorder=5)
    for key, c in [('learn_aif20', LBLUE), ('learn_prag', NAVY)]:
        axA.plot([lo, hi], [float(sn[key][0])] * 2, color=c, ls=(0, (1.2, 1.2)), lw=1.2, zorder=1)
    axA.set_xlim(0.4975, 0.57); axA.set_ylim(-0.03, 1.08)
    axA.set_xticks([0.50, 0.52, 0.54, 0.56]); axA.set_xticklabels(['0.50', '', '0.54', ''])
    axA.set_yticks([0, 0.5, 1]); axA.set_yticklabels(['0', '0.5', '1'])
    axA.set_xlabel(r'allowance $b$', labelpad=1); axA.set_ylabel('survival probability', labelpad=1)
    panel(axA, '(a)', x=0.80, y=0.30)
    # (b) survival versus preference precision inside the window, standard and ergodic buyers
    lams = el['lams']
    axB.axvspan(TH['rho_max'], LAM_A, color='#f3f3f1', lw=0, zorder=0)
    axB.axvspan(LAM_A, lams[-1] * 1.25, color='#e6effa', lw=0, zorder=0)
    axB.axvline(TH['rho_max'], color=INK2, ls='-', lw=0.6, zorder=1)
    axB.axvline(LAM_A, color=INK2, ls='-.', lw=0.7, zorder=1)
    ib = int(np.argmin(np.abs(sn['Ns'] - 200)))
    axB.axhline(float(sn['bo'][ib]), color=INK, ls=(0, (3, 2)), lw=1.2, zorder=2)
    axB.axhline(float(el['prag']), color=NAVY, ls='-', lw=1.1, zorder=2)
    axB.plot(lams, el['aif'], color=BLUE, ls='-', marker='o', mfc='white', ms=4.0, zorder=4)
    axB.plot(lams, el['erg'], color=MAGENTA, ls='-', marker='s', mfc='white', ms=4.0, zorder=5)
    axB.set_xscale('log'); axB.set_xlim(lams[0] * 0.8, lams[-1] * 1.25); axB.set_ylim(-0.03, 1.08)
    axB.set_xticks([1, 10, 100, 1000]); axB.set_xticklabels(['1', '10', r'$10^2$', r'$10^3$'])
    axB.xaxis.set_minor_formatter(mticker.NullFormatter())
    axB.set_yticks([0, 0.5, 1]); axB.set_yticklabels(['0', '0.5', '1'])
    axB.set_xlabel(r'precision $\lambda$', labelpad=1)
    axB.text(TH['rho_max'] * 0.88, 0.60, r'$\lambda_{\rm c}$', fontsize=FS, ha='right', va='center')
    axB.text(LAM_A * 1.12, 0.60, r'$\lambda_{\rm a}$', fontsize=FS, ha='left', va='center')
    panel(axB, '(b)', x=0.80, y=0.62)
    # (c) calibration with climate data, design-dependent depth
    Tg = cal['T']; m = (Tg >= 1.60) & (Tg <= 1.80)
    axK.axvspan(float(cal['T_lo']), float(cal['T_hi']), color='#f3f3f1', lw=0, zorder=0)
    axK.plot(Tg[m], cal['oracle'][m], color=MUTED, ls=':', lw=1.3, drawstyle='steps-mid')
    axK.plot(Tg[m], cal['bo'][m], color=INK, ls=(0, (3, 2)), lw=1.2, drawstyle='steps-mid', zorder=6)
    for key, c, mk_, ls in [('aif1.4', BLUE, 'o', '-'), ('aif20', LBLUE, '^', '-.'), ('prag', NAVY, 'D', '-')]:
        axK.plot(Tg[m], cal[key][m], color=c, ls=ls, marker=mk_, markevery=4, mfc='white' if mk_ != 'D' else c,
                 ms=4.0, drawstyle='steps-mid')
    axK.plot(Tg[m], ce['erg20'][m], color=MAGENTA, ls='-', marker='s', markevery=(2, 4), mfc='white', ms=4.0,
             drawstyle='steps-mid', zorder=5)
    axK.axvline(1.7, color=INK2, ls='-', lw=0.6, zorder=1)
    axK.set_xlim(1.60, 1.80); axK.set_ylim(-0.03, 1.08); axK.set_xticks([1.6, 1.7, 1.8])
    axK.set_xticklabels(['1.6', '1.7', '1.8'])
    axK.set_yticks([0, 0.5, 1]); axK.set_yticklabels(['0', '0.5', '1'])
    axK.set_xlabel(r'warming level $T$ (°C)', labelpad=1); axK.set_ylabel('survival probability', labelpad=1)
    panel(axK, '(c)', x=0.80, y=0.30)
    # (d) posted prices of the allowance sale
    Hs = np.where(sl['H'] > 0, sl['H'], np.nan); bins = sl['bins']; Ts = int(sl['T'])
    imD = axC.pcolormesh(np.arange(Ts + 1) - 0.5, bins, Hs.T, cmap=cm_blue, norm=LogNorm(vmin=1e-4, vmax=1),
                         shading='flat', rasterized=True)
    axC.axhline(float(sl['xc']), color=INK, ls='--', lw=0.9)
    axC.axhline(float(sl['xp']), color=INK2, ls=':', lw=0.9)
    axC.axhline(float(sl['xm']), color=INK2, ls=':', lw=0.9)
    axC.set_xlim(0, Ts); axC.set_ylim(0.284, 0.331)
    axC.set_xticks([0, 200, 400]); axC.set_yticks([0.29, 0.30, 0.31, 0.32]); axC.set_yticklabels(['0.29', '0.30', '0.31', '0.32'])
    axC.set_xlabel(r'round $t$', labelpad=1); axC.set_ylabel(r'posted price $x_t$', labelpad=1)
    cbD = fig.colorbar(imD, cax=caxD, ticks=[1e-4, 1e-2, 1])
    cbD.ax.tick_params(labelsize=FS, width=0.6, length=2.0); cbD.ax.minorticks_off(); cbD.outline.set_linewidth(0.6)
    panel(axC, '(d)', x=0.70, y=0.62)
    hs = [Line2D([], [], color=MUTED, ls=':', lw=1.3), Line2D([], [], color=INK, ls=(0, (3, 2)), lw=1.2)]
    hs += [Line2D([], [], color=c, ls=ls, marker=mk_, mfc='white' if mk_ != 'D' else c, ms=4.0) for _, c, mk_, ls, _ in sty]
    hs += [Line2D([], [], color=MAGENTA, ls='-', marker='s', mfc='white', ms=4.0)]
    labs = ['informed', 'Bayes-optimal'] + [x[4] for x in sty] + ['ergodic']
    fig.legend(hs, labs, loc='upper center', bbox_to_anchor=(0.53, 1.0), ncol=4, handlelength=1.6,
               columnspacing=0.7, handletextpad=0.3)
    fig.savefig(os.path.join(OUT, 'fig4.pdf'))
    plt.close(fig)


if __name__ == '__main__':
    which = sys.argv[2:] if len(sys.argv) > 2 else ['1', '2', '3', '4']
    for k in which:
        {'1': fig1, '2': fig2, '3': fig3, '4': fig4}[k]()
