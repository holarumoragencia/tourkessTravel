"""Reel vertical 1080x1920 "Patagonia, La Exploración" (Tourkess Travel).

Edita clips de stock (Pexels) con cortes al ritmo de la referencia, reencuadra los
horizontales siguiendo al sujeto, suma textos animados y cierra con el logo.

Uso: python3 reel_patagonia.py <clips_dir> <salida.mp4> [logo.png]
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS = 1080, 1920, 30
BG = (0x17, 0x24, 0x27)
FONT_DIR = Path("/usr/share/fonts/opentype/inter")
F_BLACK = str(FONT_DIR / "InterDisplay-Black.otf")
F_SEMI = str(FONT_DIR / "InterDisplay-SemiBold.otf")
TEXT_Y = 1060  # centro del bloque de texto: dentro de la zona segura de Reels

# Cada plano: clip, inicio, duración, keyframes de encuadre [(t, cx, cy)],
# zoom (inicio, fin), grade de ffmpeg, texto (kicker, titular).
# cx/cy = centro del recorte como fracción del ancho/alto del clip original.
SHOTS = [
    dict(clip="0004", ss=27.0, dur=1.9, track=[(0, .62, .49), (1.9, .66, .49)], zoom=(1.25, 1.35),
         grade="eq=contrast=1.06:saturation=1.15", text=(None, "¿TE IMAGINÁS…")),
    dict(clip="02", ss=2.0, dur=1.8, zoom=(1.0, 1.08),
         grade="eq=saturation=1.05", text=(None, "…LAGOS TURQUESAS?")),
    dict(clip="0003", ss=1.5, dur=1.8, track=[(0, .68, .5), (1.8, .63, .5)], zoom=(1.0, 1.04),
         grade="eq=contrast=1.05:saturation=1.2:brightness=0.02", text=(None, "NAVEGAR SUS AGUAS")),
    dict(clip="004", ss=7.6, dur=2.0, track=[(0, .45, .5), (2.0, .52, .5)], zoom=(1.08, 1.18),
         grade="eq=saturation=1.1", text=(None, "SAFARI 4X4")),
    dict(clip="06", ss=0.0, dur=2.0, zoom=(1.0, 1.1),
         grade="eq=contrast=1.08:saturation=1.1", text=("SIN VALLAS", "ARTE DE MILES DE AÑOS")),
    dict(clip="05", ss=3.0, dur=1.7, zoom=(1.1, 1.0),
         grade="eq=contrast=1.08:saturation=1.1", text=(None, "PAISAJES DE OTRO PLANETA")),
    dict(clip="001", ss=4.9, dur=2.3, track=[(0, .45, .5), (1.0, .43, .5), (1.9, .50, .5), (2.3, .64, .5)],
         zoom=(1.0, 1.0), grade="eq=contrast=1.12:brightness=0.03:saturation=1.15",
         text=(None, "DEL ATLÁNTICO AL PACÍFICO")),
    dict(clip="0010", ss=0.5, dur=2.0, track=[(0, .40, .5), (2.0, .42, .5)], zoom=(1.0, 1.05),
         grade="eq=saturation=1.1", text=(None, "SIN MULTITUDES")),
    dict(clip="011", ss=2.0, dur=1.8, zoom=(1.0, 1.06),
         grade="eq=saturation=1.05", text=(None, "ACOMPAÑADOS POR EL VIENTO")),
    dict(clip="012", ss=0.0, dur=1.9, zoom=(1.0, 1.04),
         grade="eq=saturation=1.08", text=(None, "UN VIAJE PARA COMPARTIR")),
    dict(clip="010", ss=0.0, dur=2.0, zoom=(1.0, 1.05),
         grade="eq=saturation=1.08", text=("SALIDA GRUPAL CON EL GEÓGRAFO", "PABLO SIGISMONDI"), text_y=640),
    dict(clip="0011", ss=3.0, dur=2.8, zoom=(1.0, 1.06), grade="eq=saturation=1.05", title=True),
]
END_DUR = 2.8


def ease_out_cubic(t):
    return 1 - (1 - t) ** 3


def clamp01(x):
    return max(0.0, min(1.0, x))


def probe_size(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout.strip()
    w, h = out.split(",")
    return int(w), int(h)


def interp(track, t):
    """Interpola linealmente (cx, cy) en los keyframes del plano."""
    if t <= track[0][0]:
        return track[0][1:]
    for (t0, x0, y0), (t1, x1, y1) in zip(track, track[1:]):
        if t <= t1:
            k = (t - t0) / (t1 - t0)
            k = k * k * (3 - 2 * k)  # smoothstep: paneos sin tirones
            return x0 + (x1 - x0) * k, y0 + (y1 - y0) * k
    return track[-1][1:]


def read_shot(clips_dir, shot):
    """Decodifica el tramo del clip a una altura suficiente para el zoom máximo."""
    path = clips_dir / f"{shot['clip']}.mp4"
    sw, sh = probe_size(path)
    zmax = max(shot["zoom"])
    # Altura de trabajo: la que deja la ventana final en >= 1920 px (sin pasar la nativa x1.0).
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
    img = Image.fromarray(frame)
    return img.resize((W, H), Image.BICUBIC, box=(x0, y0, x0 + ww, y0 + wh))


def font(path, size):
    return ImageFont.truetype(path, size)


def fit_lines(text, path, size, max_w):
    """Parte el titular en líneas que entren en max_w, achicando si hace falta."""
    while True:
        f = font(path, size)
        words, lines, cur = text.split(), [], ""
        for w in words:
            test = (cur + " " + w).strip()
            if f.getlength(test) <= max_w or not cur:
                cur = test
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
        if len(lines) <= 2 and all(f.getlength(l) <= max_w for l in lines):
            return f, lines
        size -= 6


def draw_tracked(draw, xy, text, f, fill, tracking):
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=f, fill=fill)
        x += f.getlength(ch) + tracking


def tracked_len(text, f, tracking):
    return sum(f.getlength(c) for c in text) + tracking * (len(text) - 1)


def text_layer(kicker, headline, t, text_y=TEXT_Y):
    """Capa RGBA con el kicker y el titular animados palabra por palabra."""
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    f, lines = fit_lines(headline, F_BLACK, 112, 940)
    lh = int(f.size * 1.08)
    kf = font(F_SEMI, 38)
    k_h = 70 if kicker else 0
    block_h = k_h + lh * len(lines)
    y = text_y - block_h // 2
    delay, step, dur, rise = 0.10, 0.07, 0.38, 40
    i = 0
    if kicker:
        p = ease_out_cubic(clamp01((t - delay) / dur))
        kw = tracked_len(kicker, kf, 6)
        draw_tracked(d, ((W - kw) / 2, y + rise * (1 - p)), kicker, kf, (255, 255, 255, int(235 * p)), 6)
        y += k_h
        i += 1
    for line in lines:
        words = line.split()
        lw = f.getlength(line)
        x = (W - lw) / 2
        for w in words:
            p = ease_out_cubic(clamp01((t - delay - step * i) / dur))
            d.text((x, y + rise * (1 - p)), w, font=f, fill=(255, 255, 255, int(255 * p)))
            x += f.getlength(w + " ")
            i += 1
        y += lh
    return layer


def with_shadow(base, layer, strength=0.6):
    """Compone la capa de texto con una sombra difusa para leer sobre cualquier fondo."""
    alpha = layer.split()[3]
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    shadow.putalpha(alpha.filter(ImageFilter.GaussianBlur(18)).point(lambda v: int(v * strength)))
    base = base.convert("RGBA")
    base.alpha_composite(shadow)
    base.alpha_composite(layer)
    return base


def scrim(base, center_y, half_h, strength):
    """Oscurece suavemente una banda horizontal detrás del texto."""
    ys = np.arange(H, dtype=np.float32)
    a = np.clip(1 - np.abs(ys - center_y) / half_h, 0, 1) ** 1.5 * strength
    arr = np.asarray(base.convert("RGB"), np.float32)
    arr *= (1 - a)[:, None, None]
    return Image.fromarray(arr.astype(np.uint8))


def title_layer(t):
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    items = [  # (texto, fuente, tamaño, tracking, y, retardo)
        ("SALIDA GRUPAL · 30 NOV 2026", F_SEMI, 40, 6, 760, 0.10),
        ("PATAGONIA", F_BLACK, 168, 2, 830, 0.22),
        ("LA EXPLORACIÓN", F_SEMI, 70, 14, 1030, 0.34),
        ("10 DÍAS · 9 NOCHES", F_SEMI, 44, 8, 1170, 0.50),
    ]
    for text, path, size, tr, y, dl in items:
        f = font(path, size)
        p = ease_out_cubic(clamp01((t - dl) / 0.45))
        tw = tracked_len(text, f, tr)
        draw_tracked(d, ((W - tw) / 2, y + 40 * (1 - p)), text, f, (255, 255, 255, int(255 * p)), tr)
    # Línea fina que se abre bajo el título.
    p = ease_out_cubic(clamp01((t - 0.40) / 0.5))
    half = 160 * p
    d.rectangle((W / 2 - half, 1130, W / 2 + half, 1133), fill=(255, 255, 255, int(220 * p)))
    return layer


def end_frames(logo_path, n):
    logo = Image.open(logo_path).convert("RGBA")
    logo = logo.crop(logo.getbbox())
    lw = 760
    logo = logo.resize((lw, int(logo.height * lw / logo.width)), Image.LANCZOS)
    kf, cf = font(F_SEMI, 40), font(F_BLACK, 64)
    for i in range(n):
        t = i / FPS
        canvas = Image.new("RGBA", (W, H), BG + (255,))
        p = ease_out_cubic(clamp01(t / 0.6))
        s = 0.92 + 0.08 * p
        lg = logo.resize((int(logo.width * s), int(logo.height * s)), Image.BICUBIC)
        lg.putalpha(lg.split()[3].point(lambda v: int(v * p)))
        canvas.alpha_composite(lg, ((W - lg.width) // 2, 640 - lg.height // 2))
        d = ImageDraw.Draw(canvas)
        for text, f, tr, y, dl in [("CUPOS LIMITADOS", kf, 8, 1000, 0.45), ("¡ESCRIBINOS!", cf, 2, 1060, 0.6)]:
            q = ease_out_cubic(clamp01((t - dl) / 0.4))
            tw = tracked_len(text, f, tr)
            draw_tracked(d, ((W - tw) / 2, y + 30 * (1 - q)), text, f, (255, 255, 255, int(255 * q)), tr)
        yield canvas.convert("RGB")


def main():
    clips_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    logo_path = sys.argv[3] if len(sys.argv) > 3 else Path(__file__).parent / "assets/logo_tourkess.png"
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow",
                            "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
                           stdin=subprocess.PIPE)
    for shot in SHOTS:
        frames = read_shot(clips_dir, shot)
        n = len(frames)
        track = shot.get("track", [(0, .5, .5)])
        z0, z1 = shot["zoom"]
        for i, fr in enumerate(frames):
            t = i / FPS
            k = i / max(n - 1, 1)
            cx, cy = interp(track, t)
            img = reframe(fr, cx, cy, z0 + (z1 - z0) * k)
            if shot.get("title"):
                img = scrim(img, H / 2, H / 2, 0.55)
                img = with_shadow(img, title_layer(t), 0.5)
            else:
                ty = shot.get("text_y", TEXT_Y)
                img = scrim(img, ty, 420, 0.38)
                img = with_shadow(img, text_layer(*shot["text"], t, ty))
            enc.stdin.write(img.convert("RGB").tobytes())
        print(f"{shot['clip']}: {n} cuadros", flush=True)
    for img in end_frames(logo_path, int(END_DUR * FPS)):
        enc.stdin.write(img.tobytes())
    enc.stdin.close()
    enc.wait()
    print(out)


if __name__ == "__main__":
    main()
