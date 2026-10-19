"""Reel vertical 1080x1920 "Patagonia, La Exploración" (Tourkess Travel).

Edita clips de stock (Pexels): reencuadra los horizontales siguiendo al sujeto, suma
textos animados centrados, transiciones cinematográficas (fundido a negro y
encadenados), viñeta suave y cierra con el logo.

Uso: python3 reel_patagonia.py <clips_dir> <salida.mp4> [logo.png]
"""
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS = 1080, 1920, 30
BG = (0x17, 0x24, 0x27)
HERE = Path(__file__).parent
FONT = str(HERE / "assets/fonts/Montserrat-VF.ttf")  # sustituto libre de Basic Gothic Pro
TEXT_Y = H // 2  # todo el copy centrado en pantalla

# Estilos de texto: (peso, tamaño, tracking en em)
DEST = (600, 96, 0.14)   # destino / afirmaciones en mayúscula
SUB = (500, 56, 0.0)     # titular en modo subtítulo
KICK = (400, 30, 0.04)   # texto chico arriba del titular

# Transición de entrada de cada plano: ("cut",) | ("black", seg) | ("dissolve", seg).
# track = keyframes de encuadre [(t, cx, cy)] como fracción del clip original.
SHOTS = [
    dict(clip="0004", ss=27.0, dur=2.2, track=[(0, .62, .49), (2.2, .66, .49)], zoom=(1.25, 1.35),
         grade="eq=contrast=1.04:saturation=1.05", text=dict(dest="PATAGONIA"), trans=("black", 0.5)),
    dict(clip="02", ss=2.0, dur=2.2, zoom=(1.0, 1.06),
         grade="eq=saturation=0.98", text=dict(sub="Como nunca te la contaron"), trans=("dissolve", 0.45)),
    dict(clip="0003", ss=1.5, dur=2.0, track=[(0, .70, .30), (2.0, .66, .30)], zoom=(1.15, 1.18),
         grade="eq=contrast=1.04:saturation=1.08:brightness=0.02", text=dict(sub="Navegar sus aguas"),
         trans=("cut",)),
    dict(clip="004", ss=7.8, dur=2.2, track=[(0, .46, .90), (2.2, .52, .90)], zoom=(1.18, 1.26),
         grade="eq=saturation=1.02", text=dict(sub="Safari 4x4"), trans=("cut",)),
    dict(clip="06", ss=0.0, dur=2.4, zoom=(1.0, 1.08),
         grade="eq=contrast=1.05:saturation=1.0", text=dict(sub="Arte de miles de años"), trans=("black", 0.3)),
    dict(clip="003", ss=1.0, dur=2.2, track=[(0, .70, .5), (2.2, .72, .5)], zoom=(1.0, 1.04),
         grade="eq=saturation=0.98", text=dict(dest="TODO ORGANIZADO"), trans=("dissolve", 0.5)),
    dict(clip="001", ss=4.9, dur=2.3, track=[(0, .45, .5), (1.0, .43, .5), (1.9, .50, .5), (2.3, .64, .5)],
         zoom=(1.0, 1.0), grade="eq=contrast=1.1:brightness=0.03:saturation=1.08",
         text=dict(sub="Vos solo disfrutá"), trans=("cut",)),
    dict(clip="0010", ss=0.5, dur=2.2, track=[(0, .40, .5), (2.2, .42, .5)], zoom=(1.0, 1.05),
         grade="eq=saturation=1.02", text=dict(sub="Sin multitudes"), trans=("black", 0.3)),
    dict(clip="011", ss=2.0, dur=2.0, zoom=(1.0, 1.06),
         grade="eq=saturation=1.0", text=dict(sub="Acompañados por el viento"), trans=("dissolve", 0.5)),
    dict(clip="012", ss=0.0, dur=2.0, track=[(0, .5, .2)], zoom=(1.1, 1.13),
         grade="eq=saturation=1.0", text=dict(sub="Un viaje para compartir"), trans=("cut",)),
    dict(clip="010", ss=0.0, dur=2.3, track=[(0, .5, .62)], zoom=(1.2, 1.26), grade="eq=saturation=1.0",
         text=dict(kicker="Salida grupal con el geógrafo", sub="Pablo Sigismondi"), trans=("dissolve", 0.4)),
    dict(clip="0011", ss=3.0, dur=3.0, zoom=(1.0, 1.05), grade="eq=saturation=0.98", title=True,
         trans=("black", 0.35)),
]
END = dict(dur=3.0, trans=("black", 0.45))
TEXT_OUT = 0.3  # el texto se desvanece en los últimos segundos de cada plano


def ease_out_cubic(t):
    return 1 - (1 - t) ** 3


def ease_in_out(t):
    return t * t * (3 - 2 * t)


def clamp01(x):
    return max(0.0, min(1.0, x))


@lru_cache(maxsize=None)
def font(weight, size):
    f = ImageFont.truetype(FONT, size)
    f.set_variation_by_axes([weight])
    return f


def probe_size(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout.strip()
    w, h = out.split(",")
    return int(w), int(h)


def interp(track, t):
    """Interpola (cx, cy) entre keyframes con smoothstep para paneos sin tirones."""
    if t <= track[0][0]:
        return track[0][1:]
    for (t0, x0, y0), (t1, x1, y1) in zip(track, track[1:]):
        if t <= t1:
            k = ease_in_out((t - t0) / (t1 - t0))
            return x0 + (x1 - x0) * k, y0 + (y1 - y0) * k
    return track[-1][1:]


def read_shot(clips_dir, shot):
    """Decodifica el tramo del clip a una altura suficiente para el zoom máximo."""
    path = clips_dir / f"{shot['clip']}.mp4"
    sw, sh = probe_size(path)
    zmax = max(shot["zoom"])
    th = int(min(sh, H * zmax) if sh >= H else H * zmax)
    th += th % 2
    tw = int(round(sw * th / sh))
    tw += tw % 2
    if tw * H < th * W:  # clip más angosto que 9:16: llenar el ancho
        tw = int(W * zmax)
        tw += tw % 2
        th = int(round(sh * tw / sw))
        th += th % 2
    n = int(round(shot["dur"] * FPS))
    vf = f"fps={FPS},scale={tw}:{th}:flags=lanczos,{shot['grade']}"
    cmd = ["ffmpeg", "-v", "error", "-ss", str(shot["ss"]), "-i", str(path), "-t", str(shot["dur"] + 0.2),
           "-vf", vf, "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, th, tw, 3)
    if len(frames) < n:  # el clip se terminó: congelar el último cuadro
        frames = np.concatenate([frames, np.repeat(frames[-1:], n - len(frames), 0)])
    return frames


def reframe(frame, cx, cy, z):
    """Recorta una ventana 9:16 centrada en (cx, cy) con zoom z y la lleva a 1080x1920."""
    fh, fw = frame.shape[:2]
    wh = fh / z
    ww = wh * W / H
    if ww > fw:
        ww = fw / z
        wh = ww * H / W
    x0 = min(max(cx * fw - ww / 2, 0), fw - ww)
    y0 = min(max(cy * fh - wh / 2, 0), fh - wh)
    return Image.fromarray(frame).resize((W, H), Image.BICUBIC, box=(x0, y0, x0 + ww, y0 + wh))


def tracked_len(text, f, tracking):
    return sum(f.getlength(c) for c in text) + tracking * (len(text) - 1)


def draw_tracked(draw, x, y, text, f, alpha, tracking):
    for ch in text:
        draw.text((x, y), ch, font=f, fill=(255, 255, 255, alpha))
        x += f.getlength(ch) + tracking


def fit(text, style, max_w=900):
    """Devuelve (fuente, tracking) achicando el tamaño hasta que el texto entre en max_w."""
    weight, size, em = style
    while True:
        f = font(weight, size)
        if tracked_len(text, f, size * em) <= max_w or size <= 20:
            return f, size * em
        size -= 2


def fade_in(t, delay, dur=0.6):
    return ease_out_cubic(clamp01((t - delay) / dur))


def draw_block(d, rows, t, t_end, center_y=TEXT_Y):
    """Dibuja filas centradas [(texto, fuente, tracking, alto, retardo)] con entrada suave."""
    out = clamp01((t_end - t) / TEXT_OUT)
    y = center_y - sum(r[3] for r in rows) / 2
    for text, f, tr, h, delay in rows:
        p = fade_in(t, delay)
        a = int(255 * p * out)
        x = (W - tracked_len(text, f, tr)) / 2
        asc, desc = f.getmetrics()
        ty = y + (h - (asc + desc)) / 2 + 14 * (1 - p)
        draw_tracked(d, x, ty, text, f, a, tr)
        y += h


def text_layer(spec, t, t_end):
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    rows = []
    if "dest" in spec:
        f, tr = fit(spec["dest"], DEST)
        rows.append((spec["dest"], f, tr, int(f.size * 1.2), 0.15))
    if "sub" in spec:
        f, tr = fit(spec["sub"], SUB)
        if "kicker" in spec:  # el kicker nunca más ancho que el titular
            kf, ktr = fit(spec["kicker"], KICK, tracked_len(spec["sub"], f, tr))
            rows.append((spec["kicker"], kf, ktr, int(kf.size * 1.7), 0.15))
        rows.append((spec["sub"], f, tr, int(f.size * 1.3), 0.32 if "kicker" in spec else 0.15))
    draw_block(d, rows, t, t_end)
    return layer


def title_layer(t, t_end):
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    kf, ktr = fit("Salida grupal · 30 nov 2026", KICK)
    df, dtr = fit("PATAGONIA", DEST)
    sf, str_ = fit("La exploración", (400, 46, 0.02))
    nf, ntr = fit("10 días · 9 noches", KICK)
    rows = [
        ("Salida grupal · 30 nov 2026", kf, ktr, 60, 0.15),
        ("PATAGONIA", df, dtr, int(df.size * 1.2), 0.35),
        ("La exploración", sf, str_, 90, 0.55),
        ("10 días · 9 noches", nf, ntr, 50, 0.8),
    ]
    draw_block(d, rows, t, t_end)
    return layer


def with_shadow(base, layer, strength=0.45):
    """Compone la capa de texto con una sombra difusa y suave para leer sobre cualquier fondo."""
    alpha = layer.split()[3]
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    shadow.putalpha(alpha.filter(ImageFilter.GaussianBlur(16)).point(lambda v: int(v * strength)))
    base = base.convert("RGBA")
    base.alpha_composite(shadow)
    base.alpha_composite(layer)
    return base.convert("RGB")


def band(half_h, strength):
    """Máscara multiplicativa que oscurece suavemente una banda horizontal en el centro."""
    ys = np.arange(H, dtype=np.float32)
    a = np.clip(1 - np.abs(ys - TEXT_Y) / half_h, 0, 1) ** 1.5 * strength
    return (1 - a)[:, None, None]


def vignette(strength=0.32):
    """Viñeta elíptica delicada: el centro intacto, los bordes apenas más oscuros."""
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xs - W / 2) / (W / 2)) ** 2 + ((ys - H / 2) / (H / 2)) ** 2) / np.sqrt(2)
    m = 1 - strength * ease_in_out(np.clip((r - 0.35) / 0.65, 0, 1))
    return m[..., None]


VIGNETTE = vignette()
BAND_TEXT = band(300, 0.22)
BAND_TITLE = band(520, 0.40)


def finish(img, mask):
    arr = np.asarray(img, np.float32) * mask
    return arr


def shot_frames(clips_dir, shot):
    frames = read_shot(clips_dir, shot)
    n = len(frames)
    track = shot.get("track", [(0, .5, .5)])
    z0, z1 = shot["zoom"]
    dur = n / FPS
    # En un encadenado el texto espera a que termine el fundido, así no se pisa con el anterior.
    off = shot["trans"][1] if shot["trans"][0] == "dissolve" else 0.0
    for i, fr in enumerate(frames):
        t = i / FPS
        cx, cy = interp(track, t)
        img = reframe(fr, cx, cy, z0 + (z1 - z0) * i / max(n - 1, 1))
        if shot.get("title"):
            img = Image.fromarray(finish(img, BAND_TITLE).astype(np.uint8))
            img = with_shadow(img, title_layer(t - off, dur - off))
        else:
            img = Image.fromarray(finish(img, BAND_TEXT).astype(np.uint8))
            img = with_shadow(img, text_layer(shot["text"], t - off, dur - off))
        yield finish(img, VIGNETTE)


def end_frames(logo_path, n):
    logo = Image.open(logo_path).convert("RGBA")
    logo = logo.crop(logo.getbbox())
    lw = 640
    logo = logo.resize((lw, int(logo.height * lw / logo.width)), Image.LANCZOS)
    for i in range(n):
        t = i / FPS
        canvas = Image.new("RGBA", (W, H), BG + (255,))
        p = ease_out_cubic(clamp01((t - 0.2) / 1.0))
        s = 0.96 + 0.04 * p
        lg = logo.resize((int(logo.width * s), int(logo.height * s)), Image.BICUBIC)
        lg.putalpha(lg.split()[3].point(lambda v: int(v * p)))
        canvas.alpha_composite(lg, ((W - lg.width) // 2, (H - lg.height) // 2))
        yield finish(canvas.convert("RGB"), VIGNETTE)


def main():
    clips_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    logo_path = sys.argv[3] if len(sys.argv) > 3 else HERE / "assets/logo_tourkess.png"
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow",
                            "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
                           stdin=subprocess.PIPE)

    def write(arr):
        enc.stdin.write(np.clip(arr, 0, 255).astype(np.uint8).tobytes())

    sources = [(s["trans"], s["clip"], (lambda s=s: shot_frames(clips_dir, s)), int(round(s["dur"] * FPS)))
               for s in SHOTS]
    sources.append((END["trans"], "logo", lambda: end_frames(logo_path, int(END["dur"] * FPS)),
                    int(END["dur"] * FPS)))
    total = 0
    tail = []  # últimos cuadros del plano anterior, retenidos para el encadenado
    for idx, (trans, name, gen, n) in enumerate(sources):
        nxt = sources[idx + 1][0] if idx + 1 < len(sources) else ("black", 0.4)
        k_in = int(round(trans[1] * FPS)) if trans[0] != "cut" else 0
        k_out = int(round(nxt[1] * FPS)) if nxt[0] != "cut" else 0
        for i, arr in enumerate(gen()):
            if trans[0] == "black" and i < k_in:  # sale del negro
                arr = arr * ease_in_out((i + 1) / (k_in + 1))
            if trans[0] == "dissolve" and i < k_in and tail:  # encadenado con el plano anterior
                a = ease_in_out((i + 1) / (k_in + 1))
                arr = tail.pop(0) * (1 - a) + arr * a
                write(arr)
                total += 1
                continue
            j = n - 1 - i  # cuadros que faltan para el final del plano
            if nxt[0] == "black" and j < k_out:  # entra al negro
                arr = arr * ease_in_out(j / k_out)
            if nxt[0] == "dissolve" and j < k_out:
                tail.append(arr)
                continue
            write(arr)
            total += 1
        print(f"{name}: {n} cuadros", flush=True)
    enc.stdin.close()
    enc.wait()
    print(out, f"{total / FPS:.2f}s")


if __name__ == "__main__":
    main()
