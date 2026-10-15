"""Motion graphic del logo Tourkess Travel.

Primero aparece el copy (TOURKESS / TRAVEL) letra por letra, las palabras se
abren con suavidad y el caballito de mar se revela de la cola a la corona en
el espacio del medio. Fondo: Noche azulada #172427.

Uso: python3 render.py <logo.png> <salida_dir> [vertical|horizontal|cuadrado]
"""
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

BG = (0x17, 0x24, 0x27)
FPS = 30
DUR = 6.0

BAND = (425, 580)          # franja vertical del texto en el logo original
LEFT_END, RIGHT_START = 890, 1330
LEFT_RUNS = [(0, 101), (109, 234), (244, 365), (373, 486), (487, 699), (706, 788), (798, 880)]
RIGHT_RUNS = [(1343, 1444), (1451, 1677), (1678, 1799), (1804, 1900), (1908, 2000)]

FORMATS = {
    "vertical": (1080, 1920, 0.306),
    "horizontal": (1920, 1080, 0.306),
    "cuadrado": (1080, 1080, 0.306),
}


def ease_out_cubic(t):
    return 1 - (1 - t) ** 3


def ease_in_out_quint(t):
    return 16 * t ** 5 if t < 0.5 else 1 - (-2 * t + 2) ** 5 / 2


def ease_out_expo(t):
    return 1 if t >= 1 else 1 - 2 ** (-10 * t)


def clamp01(x):
    return max(0.0, min(1.0, x))


def prog(t, start, dur):
    return clamp01((t - start) / dur)


def split_layers(logo):
    """Devuelve (letras_izq, letras_der, caballito) como capas RGBA del tamaño del logo."""
    arr = np.array(logo)
    h, w = arr.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    in_band = (ys >= BAND[0]) & (ys < BAND[1])
    left_mask = in_band & (xs < LEFT_END)
    right_mask = in_band & (xs > RIGHT_START)
    horse_mask = ~(left_mask | right_mask)

    def layer(mask, x0=0, x1=None):
        out = arr.copy()
        m = mask.copy()
        if x1 is not None:
            m &= (xs >= x0) & (xs < x1)
        out[..., 3] = np.where(m, out[..., 3], 0)
        return out

    left = [layer(left_mask, a, b) for a, b in LEFT_RUNS]
    right = [layer(right_mask, a, b) for a, b in RIGHT_RUNS]
    horse = layer(horse_mask)
    return left, right, horse


def bbox_crop(rgba):
    alpha = rgba[..., 3]
    ys, xs = np.where(alpha > 0)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return Image.fromarray(rgba[y0:y1, x0:x1]), (x0, y0)


def scaled(img, s):
    w, h = img.size
    return img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)


def place(canvas, img, x, y, opacity=1.0, blur=0.0):
    """Pega img en (x, y) con precisión subpíxel."""
    if opacity <= 0.001:
        return
    ix, iy = math.floor(x), math.floor(y)
    fx, fy = x - ix, y - iy
    pad = 4 + int(blur * 3)
    src = Image.new("RGBA", (img.width + 2 * pad, img.height + 2 * pad), (0, 0, 0, 0))
    src.paste(img, (pad, pad))
    if fx or fy:
        src = src.transform(src.size, Image.AFFINE, (1, 0, -fx, 0, 1, -fy), resample=Image.BICUBIC)
    if blur > 0.05:
        src = src.filter(ImageFilter.GaussianBlur(blur))
    if opacity < 1:
        a = np.array(src)
        a[..., 3] = (a[..., 3].astype(np.float32) * opacity).astype(np.uint8)
        src = Image.fromarray(a)
    canvas.alpha_composite(src, (ix - pad, iy - pad))


def reveal_up(img, p, feather):
    """Máscara que descubre la imagen de abajo hacia arriba (p: 0..1)."""
    if p >= 1:
        return img
    h = img.height
    edge = (1 - p) * (h + feather) - feather  # fila del borde superior visible
    rows = np.arange(h, dtype=np.float32)
    m = np.clip((rows - edge) / feather, 0, 1)
    a = np.array(img)
    a[..., 3] = (a[..., 3].astype(np.float32) * m[:, None]).astype(np.uint8)
    return Image.fromarray(a)


def render(logo_path, out_dir, fmt):
    W, H, S = FORMATS[fmt]
    logo = Image.open(logo_path).convert("RGBA")
    left, right, horse = split_layers(logo)

    def prep(rgba):
        img, (x0, y0) = bbox_crop(rgba)
        return scaled(img, S), (x0 * S, y0 * S)

    left = [prep(l) for l in left]
    right = [prep(r) for r in right]
    horse_img, horse_pos = prep(horse)

    lw, lh = logo.size
    ox, oy = (W - lw * S) / 2, (H - lh * S) / 2  # origen del logo final centrado

    # Al inicio las palabras están juntas en el centro; luego se abren a su lugar.
    gap_final = (RIGHT_RUNS[0][0] - LEFT_RUNS[-1][1]) * S
    gap_start = 46 * S
    close_shift = (gap_final - gap_start) / 2

    frames_dir = Path(out_dir) / f"frames_{fmt}"
    frames_dir.mkdir(parents=True, exist_ok=True)
    n = int(DUR * FPS)
    letters = [(img, pos, -1) for img, pos in left] + [(img, pos, 1) for img, pos in right]
    order = list(range(len(letters)))
    # orden de aparición: de afuera hacia el centro, alternando palabras
    stagger = {}
    li, ri = 0, len(left)
    k = 0
    while li < len(left) or ri < len(letters):
        if li < len(left):
            stagger[li] = k; li += 1; k += 1
        if ri < len(letters):
            stagger[ri] = k; ri += 1; k += 1

    for f in range(n):
        t = f / FPS
        canvas = Image.new("RGBA", (W, H), BG + (255,))

        # 1) Copy: cada letra sube 18px, se desenfoca menos y gana opacidad.
        open_p = ease_in_out_quint(prog(t, 1.75, 1.15))
        shift = close_shift * (1 - open_p)
        # leve respiración de escala global hacia el final
        drift = 1.0
        for i in order:
            img, (px, py), side = letters[i]
            p = ease_out_cubic(prog(t, 0.35 + stagger[i] * 0.075, 0.9))
            dx = -side * shift
            dy = (1 - p) * 22 * S / 0.46
            place(canvas, img, ox + px + dx, oy + py + dy, opacity=p, blur=(1 - p) * 6 * S / 0.46)

        # 2) Caballito: se revela de la cola a la corona mientras sube apenas.
        hp = prog(t, 2.55, 1.9)
        if hp > 0:
            e = ease_out_expo(hp) if hp < 1 else 1
            e2 = ease_out_cubic(hp)
            himg = reveal_up(horse_img, e2, feather=max(8, horse_img.height * 0.18))
            dy = (1 - e) * 30 * S / 0.46
            place(canvas, himg, ox + horse_pos[0], oy + horse_pos[1] + dy,
                  opacity=clamp01(hp * 2.2), blur=(1 - e2) * 3 * S / 0.46)

        # 3) Brillo sutil que recorre el logo al final.
        sp = prog(t, 4.35, 1.1)
        if 0 < sp < 1:
            arr = np.array(canvas).astype(np.float32)
            xs = np.arange(W, dtype=np.float32)
            ys = np.arange(H, dtype=np.float32)
            cx = -0.3 * W + sp * 1.6 * W
            d = (xs[None, :] - cx) + (ys[:, None] - H / 2) * 0.35
            glow = np.exp(-(d / (W * 0.07)) ** 2) * 0.55
            lum = (arr[..., :3].mean(-1) - 40) / 215
            mask = np.clip(lum, 0, 1) * glow
            gold = np.array([255, 255, 128], dtype=np.float32)
            arr[..., :3] += mask[..., None] * (gold - arr[..., :3])
            canvas = Image.fromarray(arr.clip(0, 255).astype(np.uint8))

        canvas.convert("RGB").save(frames_dir / f"{f:04d}.png")

    out = Path(out_dir) / f"tourkess_motion_{fmt}.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-framerate", str(FPS), "-i", str(frames_dir / "%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "slow",
        "-movflags", "+faststart", str(out),
    ], check=True)
    return out


if __name__ == "__main__":
    logo_path, out_dir = sys.argv[1], sys.argv[2]
    fmts = sys.argv[3:] or ["vertical"]
    for fmt in fmts:
        print(render(logo_path, out_dir, fmt))
