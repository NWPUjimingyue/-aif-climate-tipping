"""Vector brain glyph (side view, front to the left) for the factor graph of Fig. 1(a).

The gyri are the level set of a labyrinthine Swift-Hohenberg pattern (fixed seed),
clipped to a cerebrum outline, so the glyph is deterministic and needs no image files."""
import numpy as np
from matplotlib.patches import Polygon


def _bez(p, n=40):
    p = np.asarray(p, float)
    t = np.linspace(0, 1, n)[:, None]
    if len(p) == 3:
        return (1 - t) ** 2 * p[0] + 2 * (1 - t) * t * p[1] + t ** 2 * p[2]
    return (1 - t) ** 3 * p[0] + 3 * (1 - t) ** 2 * t * p[1] + 3 * (1 - t) * t ** 2 * p[2] + t ** 3 * p[3]


def _labyrinth(n=160, wavelength=16.0, steps=12, seed=5, bw=0.35):
    """Band-pass Gaussian noise (annulus at k0 = 2 pi / wavelength) relaxed by a few Swift-Hohenberg steps,
    u_t = r u - (k0^2 + lap)^2 u / k0^4 - u^3, on an n x n periodic grid. Its zero set is a meandering maze."""
    rng = np.random.default_rng(seed)
    k0 = 2 * np.pi / wavelength
    k = 2 * np.pi * np.fft.fftfreq(n)
    KX, KY = np.meshgrid(k, k)
    K = np.hypot(KX, KY)
    u = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((n, n))) * np.exp(-((K - k0) / (bw * k0)) ** 2)))
    u *= 0.6 / u.std()
    L = 0.6 - (k0 ** 2 - K ** 2) ** 2 / k0 ** 4
    dt = 0.3
    for _ in range(steps):
        u = np.real(np.fft.ifft2((np.fft.fft2(u) - dt * np.fft.fft2(u ** 3)) / (1 - dt * L)))
    return u


def _outline(n=600):
    """Cerebrum outline in unit coordinates (width ~1.04, height ~0.72), front at negative x:
    frontal pole, dome, occipital pole, underside, temporal lobe, temporal pole and orbital notch."""
    from scipy.interpolate import splprep, splev
    pts = np.array([(-0.50, 0.06), (-0.46, 0.25), (-0.32, 0.40), (-0.12, 0.47), (0.10, 0.47), (0.29, 0.41),
                    (0.44, 0.27), (0.52, 0.06), (0.48, -0.08), (0.34, -0.14), (0.16, -0.17), (-0.04, -0.24),
                    (-0.20, -0.23), (-0.31, -0.15), (-0.37, -0.09), (-0.47, -0.05)])
    pts = np.vstack([pts, pts[:1]])
    tck, _ = splprep([pts[:, 0], pts[:, 1]], s=0, per=1)
    x, y = splev(np.linspace(0, 1, n), tck)
    return np.c_[x, y]


def draw_brain(ax, cx, cy, w, h, light='#e3effd', mid='#bcd7f7', deep='#8db8ee', gyr2='#d3e5fa', sulc='#2f6fc0', ec='#1f5fae',
               lw=0.7, zorder=5, n=160, wavelength=18.0, fold_lw=0.45):
    """Brain of width w and height h centred near (cx, cy), in data units.
    Cerebrum with labyrinthine gyri (two shades), cerebellum with folia, and brain stem."""
    P = lambda q: np.c_[cx + np.asarray(q)[:, 0] * w, cy + np.asarray(q)[:, 1] * h]
    # brain stem and cerebellum (behind the cerebrum)
    stem = np.vstack([_bez([(0.06, -0.12), (0.09, -0.30), (0.07, -0.46)], 24),
                      _bez([(0.16, -0.46), (0.17, -0.30), (0.22, -0.10)], 24)])
    ax.add_patch(Polygon(P(stem), closed=True, fc=mid, ec=ec, lw=lw, zorder=zorder - 0.6, joinstyle='round'))
    tt = np.linspace(0, 2 * np.pi, 200)
    ang = np.deg2rad(-12)
    cb = np.c_[0.18 * np.cos(tt), 0.11 * np.sin(tt)] @ np.array([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]]) + [0.32, -0.21]
    cbp = Polygon(P(cb), closed=True, fc=mid, ec=ec, lw=lw, zorder=zorder - 0.5, joinstyle='round')
    ax.add_patch(cbp)
    for j in range(5):
        xx = np.linspace(0.12, 0.52, 50)
        yy = -0.12 - 0.038 * j - 0.17 * (xx - 0.32) + 0.005 * np.sin(np.linspace(0, 6 * np.pi, 50))
        ln, = ax.plot(*P(np.c_[xx, yy]).T, color=ec, lw=max(0.5, 0.6 * lw), zorder=zorder - 0.4, solid_capstyle='round')
        ln.set_clip_path(cbp)
    # cerebrum: gyri in two light shades (the two phases of a labyrinth u), sulci on its zero set
    out = _outline()
    cer = Polygon(P(out), closed=True, fc=light, ec='none', zorder=zorder)
    ax.add_patch(cer)
    u = _labyrinth(n=n, wavelength=wavelength)
    gx = np.linspace(-0.56, 0.56, n); gy = np.linspace(-0.30, 0.52, n)
    X, Y = np.meshgrid(cx + gx * w, cy + gy * h)
    cf = ax.contourf(X, Y, u, levels=[u.min() - 1, 0.0], colors=[gyr2], zorder=zorder + 0.05, antialiased=True)
    cs = ax.contour(X, Y, u, levels=[0.0], colors=[sulc], linewidths=fold_lw, linestyles='solid', zorder=zorder + 0.15)
    for art in (cf, cs):
        art.set_clip_path(cer)
    # soft shading: a smooth image, darker toward the base and back, lighter on the upper front
    from matplotlib.colors import to_rgb
    gx2 = np.linspace(-0.56, 0.56, 220); gy2 = np.linspace(-0.30, 0.52, 160)
    XX, YY = np.meshgrid(gx2, gy2)
    shade = np.clip(0.55 * (0.30 - YY) + 0.25 * XX, 0, None) ** 1.3
    glow = np.exp(-(((XX + 0.18) / 0.26) ** 2 + ((YY - 0.30) / 0.12) ** 2))
    rgba = np.zeros(XX.shape + (4,))
    rgba[..., :3] = to_rgb(deep)
    rgba[..., 3] = np.clip(0.75 * shade, 0, 0.45)
    rgba[..., :3] = rgba[..., :3] * (1 - glow[..., None]) + glow[..., None]
    rgba[..., 3] = np.maximum(rgba[..., 3], 0.55 * glow)
    im = ax.imshow(rgba, extent=(cx - 0.56 * w, cx + 0.56 * w, cy - 0.30 * h, cy + 0.52 * h), origin='lower',
                   interpolation='bilinear', zorder=zorder + 0.1, aspect='auto')
    im.set_clip_path(cer)
    # lateral (Sylvian) sulcus, which sets off the temporal lobe
    c = _bez([(-0.36, -0.10), (-0.14, 0.02), (0.06, -0.06), (0.30, 0.05)])
    ln, = ax.plot(*P(c).T, color=ec, lw=1.2 * lw, zorder=zorder + 0.25, solid_capstyle='round')
    ln.set_clip_path(cer)
    ax.add_patch(Polygon(P(out), closed=True, fc='none', ec=ec, lw=lw, zorder=zorder + 0.3, joinstyle='round'))
