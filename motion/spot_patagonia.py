"""Spot horizontal 1920x1080 de 10 s "Patagonia, La Exploración" (Tourkess Travel).

Directo, al estilo de la referencia (Despegar): tomas en color natural con viñeta sutil,
textos grandes y centrados, barridos diagonales y cierre con logo, fecha y WhatsApp.

Uso: python3 spot_patagonia.py <clips_dir> <salida.mp4> [musica.mp3]
"""
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS = 1920, 1080, 30
HERE = Path(__file__).parent
FONTS = HERE / "assets/fonts"
SERIF_B = str(FONTS / "InriaSerif-Bold.ttf")
SERIF = str(FONTS / "InriaSerif-Regular.ttf")
SANS = str(FONTS / "Montserrat-VF.ttf")  # sustituto de Basic Gothic Pro

# Paleta Tourkess
AZUL = (0x00, 0x4B, 0x92)
TURQUESA = (0x25, 0xA4, 0xC7)
BLANCO = (0xFF, 0xFE, 0xF8)
AMARILLO = (0xFF, 0xFF, 0x80)
AGUA = (0x7B, 0xF0, 0xB4)

WIPE = 0.40  # duración del barrido diagonal entre planos

# Los barridos caen en los golpes de la música (1,95 s, 3,87 s y 5,98 s tras el loop de un compás).
MUSIC_BAR = (0.035, 1.951)  # compás que se repite una vez para estirar el tema a 10 s

# Cada fila de texto: (texto, fuente, tamaño, peso, tracking em, color, alto de fila, retardo)
SHOTS = [
    dict(clip="glaciar", ss=0.5, dur=2.15, track=[(0, .50, .5), (2.15, .50, .5)], zoom=(1.0, 1.05), rows=[
        ("PATAGONIA", SERIF_B, 170, None, 0.06, BLANCO, 190, 0.10),
        ("La exploración", SANS, 52, 600, 0.12, BLANCO, 80, 0.35)  # blanco: el turquesa agua no contrasta sobre hielo,
    ]),
    dict(clip="02", ss=2.0, dur=2.32, track=[(0, .5, .52)], zoom=(1.0, 1.04), rows=[
        ("Salida grupal", SANS, 46, 500, 0.10, BLANCO, 80, 0.10 + WIPE),
        ("30 de noviembre 2026", SERIF_B, 120, None, 0.0, AMARILLO, 140, 0.25 + WIPE),
        ("10 días · 9 noches", SANS, 38, 500, 0.10, BLANCO, 70, 0.45 + WIPE),
    ]),
    dict(clip="001", ss=4.4, dur=2.51, track=[(0, .45, .5), (1.8, .44, .5), (2.51, .52, .5)], zoom=(1.0, 1.04), gain=1.12,
         rows=[
        ("Con el geógrafo", SANS, 46, 500, 0.10, BLANCO, 80, 0.10 + WIPE),
        ("Pablo Sigismondi", SERIF_B, 120, None, 0.0, BLANCO, 140, 0.25 + WIPE),
    ]),
]
END_DUR = 4.22  # el golpe final de la música cae en la apertura del cierre


def ease_out_cubic(t):
    return 1 - (1 - t) ** 3


def ease_in_out(t):
    return t * t * (3 - 2 * t)


def clamp01(x):
    return max(0.0, min(1.0, x))


@lru_cache(maxsize=None)
def font(path, size, weight=None):
    f = ImageFont.truetype(path, size)
    if weight:
        f.set_variation_by_axes([weight])
    return f


def probe_size(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout.strip()
    w, h = out.split(",")
    return int(w), int(h)


def interp(track, t):
    if t <= track[0][0]:
        return track[0][1:]
    for (t0, x0, y0), (t1, x1, y1) in zip(track, track[1:]):
        if t <= t1:
            k = ease_in_out((t - t0) / (t1 - t0))
            return x0 + (x1 - x0) * k, y0 + (y1 - y0) * k
    return track[-1][1:]


def read_shot(clips_dir, shot):
    """Decodifica el tramo escalado para que una ventana 16:9 quede en >= 1920 px de ancho."""
    path = clips_dir / f"{shot['clip']}.mp4"
    sw, sh = probe_size(path)
    zmax = max(shot["zoom"])
    tw = int(min(sw, W * zmax) if sw >= W else W * zmax)
    tw += tw % 2
    th = int(round(sh * tw / sw))
    th += th % 2
    if th * W < tw * H:  # clip más bajo que 16:9: llenar el alto
        th = int(H * zmax)
        th += th % 2
        tw = int(round(sw * th / sh))
        tw += tw % 2
    n = int(round(shot["dur"] * FPS))
    vf = f"fps={FPS},scale={tw}:{th}:flags=lanczos"
    cmd = ["ffmpeg", "-v", "error", "-ss", str(shot["ss"]), "-i", str(path), "-t", str(shot["dur"] + 0.2),
           "-vf", vf, "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, th, tw, 3)
    if len(frames) < n:
        frames = np.concatenate([frames, np.repeat(frames[-1:], n - len(frames), 0)])
    return frames


def reframe(frame, cx, cy, z):
    fh, fw = frame.shape[:2]
    ww = fw / z
    wh = ww * H / W
    if wh > fh:
        wh = fh / z
        ww = wh * W / H
    x0 = min(max(cx * fw - ww / 2, 0), fw - ww)
    y0 = min(max(cy * fh - wh / 2, 0), fh - wh)
    img = Image.fromarray(frame).resize((W, H), Image.BICUBIC, box=(x0, y0, x0 + ww, y0 + wh))
    return np.asarray(img, np.float32)


DARK = np.array([0x00, 0x1E, 0x40], np.float32)


def vignette(strength=0.35):
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xs - W / 2) / (W / 2)) ** 2 + ((ys - H / 2) / (H / 2)) ** 2) / np.sqrt(2)
    return (1 - strength * ease_in_out(np.clip((r - 0.3) / 0.7, 0, 1)))[..., None]


VIG = vignette(0.28)  # viñeta sutil sobre las tomas


def scrim():
    """Oscurecido elíptico y suave detrás del bloque de texto, para leer sobre hielo o cielo."""
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    s = np.exp(-((ys - H / 2) / 260) ** 2) * (0.55 + 0.45 * np.exp(-((xs - W / 2) / 650) ** 2))
    return (1 - 0.42 * s)[..., None]


SCRIM = scrim()


def natural(arr):
    """Color natural, con un leve contraste extra (reemplaza al duotono azul)."""
    return np.clip((arr - 128) * 1.06 + 128, 0, 255)


def tracked_len(text, f, tracking):
    return sum(f.getlength(c) for c in text) + tracking * (len(text) - 1)


def text_layer(rows, t, center_y=H / 2):
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    y = center_y - sum(r[6] for r in rows) / 2
    for text, path, size, weight, em, color, h, delay in rows:
        f = font(path, size, weight)
        tr = size * em
        p = ease_out_cubic(clamp01((t - delay) / 0.5))
        x = (W - tracked_len(text, f, tr)) / 2
        asc, desc = f.getmetrics()
        ty = y + (h - (asc + desc)) / 2 + 30 * (1 - p)
        for ch in text:
            d.text((x, ty), ch, font=f, fill=color + (int(255 * p),))
            x += f.getlength(ch) + tr
        y += h
    return layer


def composite(base, layer, shadow=0.5):
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).convert("RGBA")
    a = layer.split()[3]
    sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sh.putalpha(a.filter(ImageFilter.GaussianBlur(10)).point(lambda v: int(min(255, v * shadow * 1.2))))
    img.alpha_composite(sh)
    img.alpha_composite(layer)
    return np.asarray(img.convert("RGB"), np.float32)


def shot_frames(clips_dir, shot):
    frames = read_shot(clips_dir, shot)
    n = len(frames)
    z0, z1 = shot["zoom"]
    for i, fr in enumerate(frames):
        t = i / FPS
        cx, cy = interp(shot["track"], t)
        arr = natural(reframe(fr, cx, cy, z0 + (z1 - z0) * i / max(n - 1, 1)) * shot.get("gain", 1.0)) * VIG * SCRIM
        yield composite(arr, text_layer(shot["rows"], t), shadow=0.7)


def end_frames(logo_path, n):
    logo = Image.open(logo_path).convert("RGBA")
    logo = logo.crop(logo.getbbox())
    lw = 560
    logo = logo.resize((lw, int(logo.height * lw / logo.width)), Image.LANCZOS)
    # Fondo: degradado sutil de azul Tourkess hacia un azul más profundo.
    ys = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    bg = np.array(AZUL, np.float32) * (1 - 0.35 * ys) + DARK * 0.35 * ys
    bg = np.broadcast_to(bg, (H, W, 3)) * vignette(0.3)
    rows = [
        ("Consultá con Favio por WhatsApp", SANS, 44, 500, 0.06, AGUA, 80, 0.30 + WIPE),
        ("+54 9 3512 07-4454", SERIF_B, 132, None, 0.02, BLANCO, 160, 0.45 + WIPE),
        ("Patagonia · Salida grupal 30 nov 2026", SANS, 38, 500, 0.08, AMARILLO, 80, 0.65 + WIPE),
    ]
    for i in range(n):
        t = i / FPS
        img = Image.fromarray(bg.astype(np.uint8)).convert("RGBA")
        p = ease_out_cubic(clamp01((t - 0.1 - WIPE) / 0.6))
        lg = logo.copy()
        lg.putalpha(lg.split()[3].point(lambda v: int(v * p)))
        img.alpha_composite(lg, ((W - lg.width) // 2, 175 + int(20 * (1 - p))))
        layer = text_layer(rows, t, center_y=690)
        img.alpha_composite(layer)
        yield np.asarray(img.convert("RGB"), np.float32)


def wipe_mask(p):
    """Máscara del barrido diagonal (1 = plano nuevo) y bandas de color en el borde."""
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    slant = 0.45 * H  # inclinación de la diagonal
    edge = -slant - 160 + (W + 2 * slant + 320) * ease_in_out(p)
    pos = xs + (ys - H / 2) * (slant / (H / 2))
    m = (pos < edge).astype(np.float32)
    band1 = ((pos >= edge) & (pos < edge + 70)).astype(np.float32)
    band2 = ((pos >= edge + 70) & (pos < edge + 110)).astype(np.float32)
    return m[..., None], band1[..., None], band2[..., None]


def main():
    clips_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    music = sys.argv[3] if len(sys.argv) > 3 else None
    logo_path = HERE / "assets/logo_tourkess.png"
    audio_in, audio_out = [], []
    if music:
        a0, a1 = MUSIC_BAR
        audio_in = ["-i", music]
        audio_out = ["-filter_complex",
                     f"[1:a]asplit[x][y];[x]atrim=0:{a1},asetpts=PTS-STARTPTS[a];"
                     f"[y]atrim={a0},asetpts=PTS-STARTPTS[b];[a][b]acrossfade=d=0.02[m];"
                     f"[m]atrim=0:10,afade=t=out:st=9.4:d=0.6[aout]",
                     "-map", "0:v", "-map", "[aout]", "-c:a", "aac", "-b:a", "192k"]
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", *audio_in, *audio_out,
                            "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart", "-t", "10", str(out)],
                           stdin=subprocess.PIPE)
    k = int(round(WIPE * FPS))
    gens = [lambda s=s: shot_frames(clips_dir, s) for s in SHOTS]
    gens.append(lambda: end_frames(logo_path, int(round(END_DUR * FPS))))
    lens = [int(round(s["dur"] * FPS)) for s in SHOTS] + [int(round(END_DUR * FPS))]
    total, tail = 0, []
    for idx, gen in enumerate(gens):
        n, last = lens[idx], idx == len(gens) - 1
        for i, arr in enumerate(gen()):
            if idx == 0 and i < 8:  # arranque desde negro, corto
                arr = arr * ease_in_out((i + 1) / 9)
            if tail and i < k:
                m, b1, b2 = wipe_mask((i + 1) / (k + 1))
                prev = tail.pop(0)
                arr = prev * (1 - m) + arr * m
                arr = arr * (1 - b1) + np.array(TURQUESA, np.float32) * b1
                arr = arr * (1 - b2) + np.array(AGUA, np.float32) * b2
            elif not last and i >= n - k:
                tail.append(arr)
                continue
            enc.stdin.write(np.clip(arr, 0, 255).astype(np.uint8).tobytes())
            total += 1
    enc.stdin.close()
    enc.wait()
    print(out, f"{total / FPS:.2f}s")


if __name__ == "__main__":
    main()
