#!/usr/bin/env python3
"""
Generate an animated "binary portrait -> robot -> name" SVG for a GitHub profile README.

    python make_bits.py --photo me.jpg --out bits.svg
    python make_bits.py --demo --out bits.svg          # synthetic placeholder portrait

The SVG uses SMIL animation only (no JavaScript), so GitHub renders it inside <img>.
Requires: Pillow, numpy.
"""
import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

# ---------------------------------------------------------------- layout ---
FW, FH = 58, 52            # figure grid (columns x rows of bit-characters)
CW, CH = 5.4, 8.2          # size of one character cell in px
FX, FY = 22, 22            # figure top-left on the canvas
W, H = 780, 470            # canvas
TEXT_X = 350               # where the typed text starts
CYCLE = 14.0               # seconds per loop
BANDS = 6                  # vertical sweep bands -> transition looks like a wipe
LEVELS = [(0.10, 0.22), (0.24, 0.42), (0.40, 0.62), (0.58, 0.82), (0.76, 1.00)]  # (min brightness, opacity)

PORTRAIT_COLOR = "#a78bfa"
ROBOT_COLOR = "#22d3ee"
TEXT_COLOR = "#e5e7eb"
ACCENT = "#a78bfa"
BG = "#0d1117"

TEXT_LINES = [
    ("// hello, world", "#6b7280", 13, False),
    ("name   = \"Abdelrahman Oraby\"", TEXT_COLOR, 19, True),
    ("role   = \"AI & Data Engineer\"", ROBOT_COLOR, 15, False),
    ("stack  = [\"Python\", \"Flutter\", \"Arduino\"]", TEXT_COLOR, 15, False),
    ("uni    = \"Badr University in Assiut\"", TEXT_COLOR, 15, False),
    ("ieee   = \"Head of PR, BUA Student Branch\"", TEXT_COLOR, 15, False),
    ("status = \"open for freelance\"", "#4ade80", 15, False),
]


# ------------------------------------------------------------ image -> grid ---
def portrait_grid(path, focus=0.42, invert=False, contrast=1.7, keep_bg=False):
    rgb = Image.open(path).convert("RGB")
    target = (FW * CW) / (FH * CH)
    w, h = rgb.size
    if w / h > target:
        nw = int(h * target); x0 = (w - nw) // 2
        rgb = rgb.crop((x0, 0, x0 + nw, h))
    else:
        nh = int(w / target); y0 = int(max(0, min(h - nh, h * focus - nh / 2)))
        rgb = rgb.crop((0, y0, w, y0 + nh))
    rgb = rgb.resize((FW * 8, FH * 8), Image.LANCZOS)
    a = np.asarray(rgb, dtype=np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b

    # background = sky (blue-dominant) or clouds (bright, unsaturated)
    sky = (b - r) > 0.07
    cloud = (lum > 0.60) & ((b - r) > -0.02)
    bg = Image.fromarray(((sky | cloud) * 255).astype(np.uint8))
    bg = bg.filter(ImageFilter.MedianFilter(15)).filter(ImageFilter.MaxFilter(5))
    subject = 1.0 - np.asarray(bg, dtype=np.float32) / 255.0
    subject = np.asarray(Image.fromarray((subject * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(3)),
                         dtype=np.float32) / 255.0
    subject = (subject > 0.5).astype(np.float32)

    # local contrast so eyes / glasses / beard / teeth stand out
    L = Image.fromarray((lum * 255).astype(np.uint8))
    blur = np.asarray(L.filter(ImageFilter.GaussianBlur(16)), dtype=np.float32) / 255.0
    lum = np.clip(lum + 1.7 * (lum - blur), 0, 1)
    vals = lum[subject > 0]
    lo, hi = (np.percentile(vals, 2), np.percentile(vals, 98)) if vals.size else (0, 1)
    lum = np.clip((lum - lo) / max(hi - lo, 1e-3), 0, 1)
    if invert:
        lum = 1.0 - lum
    lum = np.clip((lum - 0.5) * contrast + 0.5, 0, 1) ** 1.25
    out = 0.06 + 0.94 * lum                       # subject always shows at least faintly
    fall = np.clip(1.0 - np.clip((np.linspace(0, 1, out.shape[0]) - 0.66) / 0.34, 0, 1) * 0.65, 0, 1)[:, None]
    out = out * fall
    out = out * (1.0 if keep_bg else subject)
    img = Image.fromarray((out * 255).astype(np.uint8)).resize((FW, FH), Image.BOX)
    return np.asarray(img, dtype=np.float32) / 255.0


def demo_portrait():
    """Synthetic stand-in so the pipeline can be tested without a photo."""
    S = 600
    im = Image.new("L", (S, int(S * 1.4)), 25)
    d = ImageDraw.Draw(im)
    d.ellipse((120, 60, 480, 520), fill=190)              # head
    d.rectangle((250, 480, 350, 640), fill=150)           # neck
    d.pieslice((20, 560, 580, 1100), 180, 360, fill=120)  # shoulders
    d.pieslice((110, 30, 490, 330), 180, 360, fill=70)    # hair
    d.ellipse((205, 250, 255, 285), fill=40)
    d.ellipse((345, 250, 395, 285), fill=40)
    d.arc((230, 380, 370, 450), 20, 160, fill=60, width=8)
    return im.filter(ImageFilter.GaussianBlur(3))


def robot_grid():
    """Procedural robot head + shoulders, drawn supersampled then reduced to the grid."""
    K = 10
    im = Image.new("L", (40 * K, 36 * K), 0)
    d = ImageDraw.Draw(im)
    cx = 40 * K // 2

    def box(x0, y0, x1, y1, fill=None, outline=None, r=0, width=0):
        d.rounded_rectangle((x0 * K, y0 * K, x1 * K, y1 * K), radius=r * K,
                            fill=fill, outline=outline, width=int(width * K))

    box(cx / K - 0.5, 3, cx / K + 0.5, 7, fill=220)                  # antenna stem
    d.ellipse((cx - 2.2 * K, 0.8 * K, cx + 2.2 * K, 4.0 * K), fill=255)  # antenna ball
    box(6, 7, 34, 24, fill=22, outline=255, r=3, width=1.7)         # head (hollow)
    box(3, 12, 6, 19, fill=200, outline=255, r=1, width=1.0)        # left ear
    box(34, 12, 37, 19, fill=200, outline=255, r=1, width=1.0)      # right ear
    box(10, 10.5, 17, 15.5, fill=255, r=1)                          # left eye
    box(23, 10.5, 30, 15.5, fill=255, r=1)                          # right eye
    box(12.5, 11.8, 14.5, 14.2, fill=0, r=0.5)                      # pupils
    box(25.5, 11.8, 27.5, 14.2, fill=0, r=0.5)
    box(11, 18.5, 29, 22, fill=0, outline=230, r=1, width=0.6)      # mouth panel
    for x in range(14, 28, 3):                                      # grill
        d.line((x * K, 19.2 * K, x * K, 21.3 * K), fill=255, width=int(0.6 * K))
    box(16, 24, 24, 27.5, fill=170, outline=230, r=0.5, width=0.4)  # neck
    box(3, 27.5, 37, 36, fill=22, outline=255, r=3, width=1.7)      # shoulders
    box(16, 29.5, 24, 33, fill=255, r=1)                            # chest light
    im = im.resize((FW, FH), Image.LANCZOS)
    return np.asarray(im, dtype=np.float32) / 255.0


# ---------------------------------------------------------------- svg bits ---
def keyframes(points):
    """points: [(seconds, value)] -> (keyTimes, values) normalised to the cycle."""
    pts = sorted(points)
    out, last_t = [], -1.0
    for t, v in pts:
        t = max(0.0, min(CYCLE, t))
        if t <= last_t:
            t = last_t + 1e-3
        out.append((min(t, CYCLE), v))
        last_t = t
    if out[0][0] > 0:
        out.insert(0, (0.0, out[0][1]))
    if out[-1][0] < CYCLE:
        out.append((CYCLE, out[-1][1]))
    else:
        out[-1] = (CYCLE, out[-1][1])
    kt = ";".join(f"{t / CYCLE:.4f}" for t, _ in out)
    vals = ";".join(f"{v:g}" for _, v in out)
    return kt, vals


def animate(attr, points, extra=""):
    kt, vals = keyframes(points)
    return (f'<animate attributeName="{attr}" dur="{CYCLE:g}s" repeatCount="indefinite" '
            f'keyTimes="{kt}" values="{vals}" {extra}/>')


def glyph_rows(grid, rng, level_lo, level_hi):
    """Return per-row (xs, chars) for cells whose brightness falls in [lo, hi)."""
    rows = []
    for r in range(FH):
        xs, chars = [], []
        for c in range(FW):
            b = grid[r, c]
            if level_lo <= b < level_hi:
                xs.append(FX + c * CW)
                chars.append(rng.choice("01"))
        if xs:
            rows.append((FY + (r + 0.85) * CH, xs, "".join(chars)))
    return rows


def layer_svg(grid, color, rng, cls, t_in, t_out, flip):
    """One bit-layer, split into vertical bands that fade at staggered times (wipe effect)."""
    parts = [f'<g class="{cls}" fill="{color}" font-weight="700">']
    band_w = FW / BANDS
    for bi in range(BANDS):
        k = (BANDS - 1 - bi) if flip else bi
        d = k * 0.16
        masked = np.zeros_like(grid)
        c0, c1 = int(round(bi * band_w)), int(round((bi + 1) * band_w))
        masked[:, c0:c1] = grid[:, c0:c1]
        pts = [(0, 0), (t_in[0] + d, 0), (t_in[1] + d, 1), (t_out[0] + d, 1), (t_out[1] + d, 0), (CYCLE, 0)]
        parts.append(f'<g opacity="0">{animate("opacity", pts)}')
        for li, (lo, op) in enumerate(LEVELS):
            hi = LEVELS[li + 1][0] if li + 1 < len(LEVELS) else 9
            rows = glyph_rows(masked, rng, lo, hi)
            if not rows:
                continue
            parts.append(f'<g fill-opacity="{op}">')
            for y, xs, chars in rows:
                xl = " ".join(f"{x:.1f}" for x in xs)
                parts.append(f'<text x="{xl}" y="{y:.1f}" xml:space="preserve">{chars}</text>')
            parts.append("</g>")
        parts.append("</g>")
    parts.append("</g>")
    return "\n".join(parts)


def background_field(rng):
    rows = []
    cols = int(W / CW)
    for r in range(int(H / CH)):
        rows.append(f'<text x="0" y="{(r + 0.85) * CH:.1f}">'
                    + "".join(rng.choice("01") for _ in range(cols)) + "</text>")
    return (f'<g fill="#6e40c9" fill-opacity="0.07" font-size="{CW / 0.6:.1f}">' + "".join(rows) + "</g>")


def text_block():
    out, defs = [], []
    y = 66
    start = 7.0
    approx = lambda s, fs: len(s) * fs * 0.62
    for i, (s, color, fs, bold) in enumerate(TEXT_LINES):
        y += fs + 20 if i else 0
        t0 = start + i * 0.85
        width = approx(s, fs) + 12
        cid = f"clip{i}"
        kt_pts = [(0, 0), (t0, 0), (t0 + 0.75, width), (12.6, width), (13.6, 0), (CYCLE, 0)]
        defs.append(f'<clipPath id="{cid}"><rect x="{TEXT_X - 2}" y="{y - fs - 2}" width="0" height="{fs + 10}">'
                    f'{animate("width", kt_pts)}</rect></clipPath>')
        weight = ' font-weight="700"' if bold else ""
        out.append(f'<text xml:space="preserve" x="{TEXT_X}" y="{y}" font-size="{fs}" fill="{color}"{weight} '
                   f'clip-path="url(#{cid})">{s.replace("&", "&amp;")}</text>')
    last_y = y + 26
    cursor = (f'<rect x="{TEXT_X}" y="{last_y - 14}" width="9" height="16" fill="{ACCENT}" opacity="0">'
              f'{animate("opacity", [(0, 0), (start + len(TEXT_LINES) * 0.85, 0), (start + len(TEXT_LINES) * 0.85 + 0.01, 1), (12.6, 1), (13.4, 0), (CYCLE, 0)])}'
              f'<animate attributeName="fill-opacity" dur="0.9s" repeatCount="indefinite" values="1;1;0;0" keyTimes="0;0.5;0.5;1"/></rect>')
    return "\n".join(defs), "\n".join(out) + "\n" + cursor


def build(portrait, robot, seed=7):
    rng = random.Random(seed)
    defs, text = text_block()
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="Animated binary portrait that turns into a robot and types Abdelrahman Oraby's details">
<title>Abdo Oraby: binary portrait to robot</title>
<defs>{defs}</defs>
<style>text{{font-family:'Fira Code','JetBrains Mono',Consolas,Menlo,'DejaVu Sans Mono',monospace;white-space:pre}}</style>
<rect width="{W}" height="{H}" rx="14" fill="{BG}" stroke="#30363d"/>
{background_field(rng)}
<g font-size="{CW / 0.6:.1f}">
{layer_svg(portrait, PORTRAIT_COLOR, rng, "portrait", (0.3, 1.6), (5.0, 6.2), flip=False)}
{layer_svg(robot, ROBOT_COLOR, rng, "robot", (5.4, 6.6), (12.6, 13.6), flip=True)}
</g>
<g fill="{ACCENT}" font-size="11" opacity="0.8"><text x="{TEXT_X}" y="30">$ ./bits --render --user AbdoOraby21</text></g>
{text}
</svg>'''
    return svg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--out", default="bits.svg")
    ap.add_argument("--focus", type=float, default=0.42, help="vertical focus of crop, 0=top 1=bottom")
    ap.add_argument("--invert", action="store_true", help="use if your photo has a bright background")
    ap.add_argument("--keep-bg", action="store_true", help="do not remove sky/cloud background")
    ap.add_argument("--contrast", type=float, default=1.7)
    a = ap.parse_args()
    if not a.photo and not a.demo:
        ap.error("give --photo or --demo")
    src = demo_portrait() if a.demo else a.photo
    if a.demo:
        tmp = Path(a.out).with_suffix(".demo.png")
        src.save(tmp)
        src = tmp
    p = portrait_grid(src, a.focus, a.invert, a.contrast, a.keep_bg)
    Path(a.out).write_text(build(p, robot_grid()), encoding="utf-8")
    print(f"wrote {a.out} ({Path(a.out).stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
