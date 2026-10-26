"""Reel vertical 1080x1920 "India Deluxe, exclusiva para mujeres" (Tourkess Travel).

Mismo lenguaje que el reel de Patagonia (textos centrados, viñeta suave, fundidos y
encadenados) más los barridos diagonales con líneas de color del spot horizontal,
usados con moderación para sumar un toque publicitario sin perder lo cinematográfico.

Los videos fuente ya vienen editados (cortes cada ~0,9 s): se separan las tomas, se
ralentizan con interpolación de movimiento y se agrupan de a dos por cada texto.

Uso: python3 reel_india.py <clips_dir> <salida.mp4>  (clips: delhi, jaipur, agra, varanasi, emirates)
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import reel_patagonia as rp
from reel_patagonia import (DEST, FPS, H, KICK, SUB, VIGNETTE, W, BAND_TEXT, BAND_TITLE, clamp01, draw_block,
                            ease_in_out, ease_out_cubic, finish, fit, read_shot, reframe, text_layer, with_shadow)

HERE = Path(__file__).parent
TURQUESA = np.array([0x25, 0xA4, 0xC7], np.float32)
AGUA = np.array([0x7B, 0xF0, 0xB4], np.float32)

SPEED = 0.6  # las tomas fuente duran ~0,9 s: a 0,6x quedan en ~1,5 s, ritmo pausado

# Cada segmento: tomas fuente [(clip, desde, hasta)] que se ralentizan y se unen con corte seco,
# el texto que va encima y la transición de entrada ("cut" | "black" | "dissolve" | "lines").
SEGMENTS = [
    dict(name="c01", parts=[("delhi", 0.03, 0.95), ("delhi", 1.90, 2.82)], dur=2.8,
         text=dict(dest="INDIA"), trans=("black", 0.5)),
    dict(name="c02", parts=[("delhi", 3.82, 4.68), ("delhi", 0.99, 1.85)], dur=2.6,
         text=dict(sub="Exclusiva para mujeres"), trans=("dissolve", 0.45)),
    # Avión de Emirates: el clip ya tiene movimiento propio, va a velocidad normal.
    dict(name="c03", parts=[("emirates", 5.50, 8.30, 1.0)], dur=2.6,
         text=dict(kicker="Vuelos internacionales", sub="con Emirates"), trans=("lines", 0.5)),
    dict(name="c04", parts=[("delhi", 2.86, 3.78), ("delhi", 4.72, 5.45)], dur=2.4,
         text=dict(dest="DELHI"), trans=("dissolve", 0.45)),
    dict(name="c05", parts=[("jaipur", 1.90, 2.75), ("jaipur", 2.80, 3.72)], dur=2.8,
         text=dict(dest="JAIPUR"), trans=("cut",)),
    dict(name="c06", parts=[("agra", 0.03, 0.95), ("agra", 3.82, 4.68)], dur=2.8,
         text=dict(dest="AGRA"), trans=("lines", 0.5)),
    dict(name="c07", parts=[("varanasi", 1.54, 2.14), ("varanasi", 2.98, 3.67)], dur=2.1,
         text=dict(dest="VARANASI"), trans=("cut",)),
    dict(name="c08", parts=[("varanasi", 2.21, 2.91), ("jaipur", 0.99, 1.85)], dur=2.4,
         text=dict(kicker="Guía acompañante", sub="en español"), trans=("dissolve", 0.45)),
    dict(name="c09", parts=[("varanasi", 3.75, 4.43, 0.22)], dur=3.0, title=True, trans=("black", 0.4)),
]
# Fondo del cierre: el Taj Mahal al amanecer en cámara lenta, oscurecido para que se lea el precio.
END_BG = dict(name="c10", parts=[("agra", 4.72, 5.45, 0.2)], dur=3.6)
END = dict(dur=3.6, trans=("lines", 0.55))


def prepare(src_dir, work):
    """Ralentiza y une las tomas de cada segmento en un clip intermedio (se reutiliza si ya existe)."""
    work.mkdir(parents=True, exist_ok=True)
    for seg in SEGMENTS + [END_BG]:
        out = work / f"{seg['name']}.mp4"
        if out.exists():
            continue
        inputs, chains = [], []
        for k, part in enumerate(seg["parts"]):
            clip, a, b = part[:3]
            speed = part[3] if len(part) > 3 else SPEED
            inputs += ["-ss", str(a), "-t", str(b - a), "-i", str(src_dir / f"{clip}.mp4")]
            interp = f"minterpolate=fps={FPS}:mi_mode=mci:mc_mode=aobmc:vsbmc=1" if speed < 1 else f"fps={FPS}"
            chains.append(f"[{k}:v]setpts=(PTS-STARTPTS)/{speed},{interp},"
                          f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1[v{k}]")
        n = len(seg["parts"])
        graph = ";".join(chains) + ";" + "".join(f"[v{k}]" for k in range(n)) + f"concat=n={n}:v=1:a=0[out]"
        subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", graph, "-map", "[out]",
                        "-c:v", "libx264", "-crf", "12", "-preset", "fast", "-pix_fmt", "yuv420p", str(out)],
                       check=True)
        print("preparado", out.name, flush=True)


def lines_mask(p):
    """Barrido diagonal (1 = plano nuevo) con dos líneas finas de color en el borde."""
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    slant = 0.35 * W
    pos = xs + (ys - H / 2) * (slant / (H / 2))
    edge = -slant - 80 + (W + 2 * slant + 160) * ease_in_out(p)
    m = (pos < edge).astype(np.float32)
    b1 = ((pos >= edge) & (pos < edge + 26)).astype(np.float32)
    b2 = ((pos >= edge + 40) & (pos < edge + 50)).astype(np.float32)
    return m[..., None], b1[..., None], b2[..., None]


def accent_line(layer, t, t_end, y, delay=0.45, half=70):
    """Línea fina en turquesa agua que se abre bajo el texto (detalle gráfico del spot)."""
    p = ease_out_cubic(clamp01((t - delay) / 0.6)) * clamp01((t_end - t) / rp.TEXT_OUT)
    if p <= 0:
        return
    d = ImageDraw.Draw(layer)
    w = half * p
    d.rectangle((W / 2 - w, y, W / 2 + w, y + 3), fill=tuple(int(c) for c in AGUA) + (int(230 * p),))


def title_layer(t, t_end):
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    kf, ktr = fit("Salida grupal · 07 mar 2027", KICK)
    df, dtr = fit("INDIA DELUXE", DEST)
    sf, s_tr = fit("12 días · 11 noches", (500, 46, 0.02))
    cf, ctr = fit("Delhi · Udaipur · Jaipur · Agra · Varanasi", (400, 28, 0.04))
    rows = [
        ("Salida grupal · 07 mar 2027", kf, ktr, 60, 0.15),
        ("INDIA DELUXE", df, dtr, int(df.size * 1.25), 0.35),
        ("12 días · 11 noches", sf, s_tr, 80, 0.6),
        ("Delhi · Udaipur · Jaipur · Agra · Varanasi", cf, ctr, 60, 0.85),
    ]
    draw_block(d, rows, t, t_end)
    accent_line(layer, t, t_end, H / 2 + 26, delay=0.5, half=90)
    return layer


def seg_frames(work, seg):
    shot = dict(clip=seg["name"], ss=0.0, dur=seg["dur"], zoom=(1.0, 1.05), grade="null")
    frames = read_shot(work, shot)
    n = len(frames)
    dur = n / FPS
    off = seg["trans"][1] if seg["trans"][0] in ("dissolve", "lines") else 0.0
    for i, fr in enumerate(frames):
        t = i / FPS
        img = reframe(fr, .5, .5, 1.0 + 0.05 * i / max(n - 1, 1))
        if seg.get("title"):
            img = Image.fromarray(finish(img, BAND_TITLE).astype(np.uint8))
            img = with_shadow(img, title_layer(t - off, dur - off))
        else:
            img = Image.fromarray(finish(img, BAND_TEXT).astype(np.uint8))
            layer = text_layer(seg["text"], t - off, dur - off)
            if "dest" in seg["text"]:
                accent_line(layer, t - off, dur - off, H / 2 + 72)
            img = with_shadow(img, layer)
        yield finish(img, VIGNETTE)


def end_frames(work, n, logo_path):
    """Cierre sobre video (Taj Mahal) oscurecido: logo y precio en cuotas."""
    logo = Image.open(logo_path).convert("RGBA")
    logo = logo.crop(logo.getbbox())
    lw = 560
    logo = logo.resize((lw, int(logo.height * lw / logo.width)), Image.LANCZOS)
    frames = read_shot(work, dict(clip=END_BG["name"], ss=0.0, dur=END_BG["dur"], zoom=(1.0, 1.06), grade="null"))
    dim = rp.vignette(0.4) * 0.5
    kf, ktr = fit("5 cuotas fijas de", KICK)
    pf, ptr = fit("USD 2.630", (600, 120, 0.04))
    bf, btr = fit("Base doble", (400, 28, 0.06))
    rows = [("5 cuotas fijas de", kf, ktr, 64, 0.65), ("USD 2.630", pf, ptr, 150, 0.8),
            ("Base doble", bf, btr, 56, 1.0)]
    off = END["trans"][1]
    for i in range(n):
        t = i / FPS - off
        fr = reframe(frames[min(i, len(frames) - 1)], .5, .5, 1.0 + 0.06 * i / max(n - 1, 1))
        bg = np.asarray(fr, np.float32) * dim
        img = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8)).convert("RGBA")
        p = ease_out_cubic(clamp01((t - 0.1) / 0.7))
        lg = logo.copy()
        lg.putalpha(lg.split()[3].point(lambda v: int(v * p)))
        img.alpha_composite(lg, ((W - lg.width) // 2, 560 + int(20 * (1 - p))))
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw_block(ImageDraw.Draw(layer), rows, t, 99, center_y=1150)
        accent_line(layer, t, 99, 1305, delay=1.1, half=80)
        img = Image.fromarray(np.asarray(with_shadow(img.convert("RGB"), layer, 0.5))).convert("RGBA")
        yield np.asarray(img.convert("RGB"), np.float32)


def main():
    src_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    work = src_dir / "_segmentos"
    prepare(src_dir, work)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow",
                            "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
                           stdin=subprocess.PIPE)

    def write(arr):
        enc.stdin.write(np.clip(arr, 0, 255).astype(np.uint8).tobytes())

    sources = [(s["trans"], s["name"], (lambda s=s: seg_frames(work, s)), int(round(s["dur"] * FPS)))
               for s in SEGMENTS]
    sources.append((END["trans"], "cierre", lambda: end_frames(work, int(END["dur"] * FPS), HERE / "assets/logo_tourkess.png"),
                    int(END["dur"] * FPS)))
    total, tail = 0, []
    for idx, (trans, name, gen, n) in enumerate(sources):
        nxt = sources[idx + 1][0] if idx + 1 < len(sources) else ("black", 0.5)
        k_in = int(round(trans[1] * FPS)) if trans[0] != "cut" else 0
        k_out = int(round(nxt[1] * FPS)) if nxt[0] != "cut" else 0
        for i, arr in enumerate(gen()):
            if trans[0] == "black" and i < k_in:
                arr = arr * ease_in_out((i + 1) / (k_in + 1))
            if trans[0] in ("dissolve", "lines") and i < k_in and tail:
                prev = tail.pop(0)
                p = (i + 1) / (k_in + 1)
                if trans[0] == "dissolve":
                    a = ease_in_out(p)
                    arr = prev * (1 - a) + arr * a
                else:
                    m, b1, b2 = lines_mask(p)
                    arr = prev * (1 - m) + arr * m
                    arr = arr * (1 - b1) + TURQUESA * b1
                    arr = arr * (1 - b2) + AGUA * b2
                write(arr)
                total += 1
                continue
            j = n - 1 - i
            if nxt[0] == "black" and j < k_out:
                arr = arr * ease_in_out(j / k_out)
            if nxt[0] in ("dissolve", "lines") and j < k_out:
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
