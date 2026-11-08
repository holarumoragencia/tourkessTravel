"""Cartel de partidas tipo split-flap (paletas que caen) con 3 destinos.

Secuencia:
  - El cartel arranca con las paletas en blanco y se escribe RÍO DE JANEIRO.
  - Queda 1 s y cambia a BUZIOS (las paletas caen en orden aleatorio).
  - Queda 1 s y cambia a CABO FRÍO (cascada de derecha a izquierda).
  - Queda 2 s y el video termina con ese destino fijo.

Uso: python3 cartel_destinos.py <salida_dir> [vertical|horizontal|cuadrado|alpha ...]
  "alpha" exporta el cartel solo, con fondo transparente (.mov Animation y .webm),
  para montarlo sobre otro video en el editor.
"""
import math
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FPS = 60
BG = (0x17, 0x24, 0x27)  # Noche azulada (paleta Tourkess)

HEADER_TEXT = "DEPARTURES"
DESTINOS = [  # (destino, código de aeropuerto)
    ("RÍO DE JANEIRO", "GIG"),
    ("BUZIOS", "BZC"),
    ("CABO FRÍO", "CFB"),
]
HOLDS = [1.0, 1.0, 2.0]  # cuánto queda fijo cada destino (desde que cae la última paleta)
START = 0.30             # pausa inicial con el cartel en blanco
FLIP = 0.07              # duración de la caída de una paleta (s)

NAME_CELLS, CODE_CELLS = 15, 3
DRUM = " ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚ0123456789"

FONT_FLAP = "/usr/share/fonts/opentype/inter/Inter-SemiBold.otf"
FONT_HEAD = "/usr/share/fonts/opentype/inter/Inter-Bold.otf"
FONT_ICON = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

# Geometría en unidades de diseño (se escalan según el formato)
CW, CH, GAP, SPACER, PAD = 80, 112, 8, 44, 22
HEAD_H, ROW_GAP = 150, 22
BOARD_W = PAD * 2 + (NAME_CELLS + CODE_CELLS) * CW + (NAME_CELLS + CODE_CELLS - 2) * GAP + SPACER
ROW_H = CH + PAD * 2
BOARD_H = HEAD_H + ROW_GAP + ROW_H

FORMATS = {  # ancho, alto, ancho del cartel relativo al ancho del cuadro
    "horizontal": (1920, 1080, 0.86),
    "vertical": (1080, 1920, 0.93),
    "cuadrado": (1080, 1080, 0.93),
    "alpha": (1920, 1080, 0.86),
}


# ----------------------------------------------------------------------------
# Dibujo
# ----------------------------------------------------------------------------
def vgrad(w, h, top, bottom):
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    c = np.array(top, np.float32) * (1 - t) + np.array(bottom, np.float32) * t
    arr = np.repeat(c, w, axis=1)
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8), "RGB")


def rounded_mask(w, h, r, ss=4):
    m = Image.new("L", (w * ss, h * ss), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, w * ss - 1, h * ss - 1), r * ss, fill=255)
    return m.resize((w, h), Image.LANCZOS)


def panel(w, h, r, top, bottom, border=(70, 70, 70)):
    img = vgrad(w, h, top, bottom).convert("RGBA")
    img.putalpha(rounded_mask(w, h, r))
    # borde fino claro arriba para dar volumen
    ss = 4
    edge = Image.new("L", (w * ss, h * ss), 0)
    ImageDraw.Draw(edge).rounded_rectangle((0, 0, w * ss - 1, h * ss - 1), r * ss, outline=255, width=ss * 2)
    edge = edge.resize((w, h), Image.LANCZOS)
    fade = np.linspace(1, 0.15, h, dtype=np.float32)[:, None]
    a = (np.array(edge, np.float32) * fade * 0.55).astype(np.uint8)
    col = Image.new("RGBA", (w, h), border + (0,))
    col.putalpha(Image.fromarray(a))
    img.alpha_composite(col)
    return img


class Flaps:
    """Caras pre-renderizadas de cada carácter, partidas en mitad superior/inferior."""

    def __init__(self, s):
        self.w, self.h = round(CW * s), round(CH * s)
        self.h2 = self.h // 2
        self.r = max(2, round(7 * s))
        ss = 3
        W, H = self.w * ss, self.h * ss
        font = ImageFont.truetype(FONT_FLAP, round(CH * 0.72 * s * ss))
        cap = font.getbbox("H")
        cap_mid = (cap[1] + cap[3]) / 2
        mask = rounded_mask(self.w, self.h, self.r)
        top_bg = vgrad(self.w, self.h2, (58, 58, 58), (38, 38, 38))
        bot_bg = vgrad(self.w, self.h - self.h2, (34, 34, 34), (22, 22, 22))
        self.faces = {}
        for ch in DRUM:
            face = Image.new("RGBA", (self.w, self.h))
            face.paste(top_bg, (0, 0))
            face.paste(bot_bg, (0, self.h2))
            if ch != " ":
                txt = Image.new("L", (W, H), 0)
                d = ImageDraw.Draw(txt)
                tw = d.textlength(ch, font=font)
                d.text(((W - tw) / 2, H / 2 - cap_mid + ss), ch, font=font, fill=255)
                txt = txt.resize((self.w, self.h), Image.LANCZOS)
                ink = Image.new("RGBA", (self.w, self.h), (236, 236, 232, 0))
                ink.putalpha(txt)
                face.alpha_composite(ink)
            face.putalpha(mask)
            self.faces[ch] = (face.crop((0, 0, self.w, self.h2)),
                              face.crop((0, self.h2, self.w, self.h)))
        self.overlay = self._overlay(s)

    def _overlay(self, s):
        """Ranura central y bisagras laterales (van encima de todo)."""
        ov = Image.new("RGBA", (self.w, self.h))
        d = ImageDraw.Draw(ov)
        lh = max(1, round(2.2 * s))
        y = self.h2 - lh // 2
        d.rectangle((0, y, self.w, y + lh - 1), fill=(8, 8, 8, 255))
        d.rectangle((0, y + lh, self.w, y + lh), fill=(80, 80, 80, 90))
        pw, ph = max(2, round(5 * s)), max(3, round(16 * s))
        for x0 in (0, self.w - pw):
            d.rectangle((x0, self.h2 - ph // 2, x0 + pw - 1, self.h2 + ph // 2), fill=(12, 12, 12, 255))
        ov.putalpha(Image.fromarray(np.minimum(np.array(ov)[..., 3], np.array(rounded_mask(self.w, self.h, self.r)))))
        return ov

    @staticmethod
    def shade(img, k):
        a = np.array(img).astype(np.float32)
        a[..., :3] *= k
        return Image.fromarray(a.clip(0, 255).astype(np.uint8))

    def cell(self, a, b, p):
        """Celda cayendo de a hacia b, p en [0, 1). p=None => celda fija con a."""
        out = Image.new("RGBA", (self.w, self.h))
        if p is None:
            out.alpha_composite(self.faces[a][0], (0, 0))
            out.alpha_composite(self.faces[a][1], (0, self.h2))
        else:
            out.alpha_composite(self.shade(self.faces[b][0], 0.75 + 0.25 * p), (0, 0))
            out.alpha_composite(self.faces[a][1], (0, self.h2))
            if p < 0.5:  # la mitad superior de "a" cae hacia el frente
                sy = math.cos(p * math.pi)
                fh = max(1, round(self.h2 * sy))
                flap = self.faces[a][0].resize((self.w, fh), Image.BILINEAR)
                out.alpha_composite(self.shade(flap, 0.45 + 0.55 * sy), (0, self.h2 - fh))
            else:  # la mitad inferior de "b" termina de bajar
                sy = -math.cos(p * math.pi)
                fh = max(1, round((self.h - self.h2) * sy))
                flap = self.faces[b][1].resize((self.w, fh), Image.BILINEAR)
                # sombra que proyecta sobre la paleta de abajo
                sh = Image.new("RGBA", (self.w, self.h - self.h2), (0, 0, 0, round(110 * (1 - sy))))
                sh.putalpha(Image.fromarray(np.minimum(np.array(sh)[..., 3], np.array(self.faces[a][1])[..., 3])))
                out.alpha_composite(sh, (0, self.h2))
                out.alpha_composite(self.shade(flap, 0.55 + 0.45 * sy), (0, self.h2))
        out.alpha_composite(self.overlay)
        return out


def plane_icon(size):
    """Avión de perfil despegando (silueta dibujada a mano)."""
    ss = 4
    S = size * ss
    m = Image.new("L", (S * 2, S * 2), 0)
    d = ImageDraw.Draw(m)
    u = S / 100
    ox, oy = S * 0.5, S * 0.8

    def P(pts):
        return [(ox + x * u, oy + y * u) for x, y in pts]

    d.rounded_rectangle((ox + 6 * u, oy + 18 * u, ox + 104 * u, oy + 30 * u), 6 * u, fill=255)  # fuselaje
    d.polygon(P([(6, 24), (0, 2), (11, 2), (28, 20)]), fill=255)                              # cola
    d.polygon(P([(44, 26), (62, 26), (40, 52), (31, 52)]), fill=255)                          # ala
    d.polygon(P([(48, 20), (60, 20), (46, 6), (40, 6)]), fill=255)                            # ala lejana
    d.polygon(P([(10, 22), (24, 22), (14, 32), (6, 32)]), fill=255)                           # estabilizador
    m = m.rotate(18, resample=Image.BICUBIC, center=(ox + 50 * u, oy + 25 * u))
    m = m.crop(m.getbbox())
    return m.resize((max(1, m.width // ss), max(1, m.height // ss)), Image.LANCZOS)


def static_board(s, flaps):
    """Fondo del cartel (paneles, encabezado) y posiciones de las celdas."""
    bw, bh = round(BOARD_W * s), round(BOARD_H * s)
    margin = round(40 * s)
    img = Image.new("RGBA", (bw + margin * 2, bh + margin * 2))
    hh, rg, rh = round(HEAD_H * s), round(ROW_GAP * s), round(ROW_H * s)

    # sombra suave de los paneles
    sh = Image.new("L", img.size, 0)
    d = ImageDraw.Draw(sh)
    off = round(10 * s)
    d.rounded_rectangle((margin, margin + off, margin + bw, margin + hh + off), round(10 * s), fill=150)
    d.rounded_rectangle((margin, margin + hh + rg + off, margin + bw, margin + bh + off), round(10 * s), fill=150)
    sh = sh.filter(ImageFilter.GaussianBlur(18 * s))
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    shadow.putalpha(sh)
    img.alpha_composite(shadow)

    img.alpha_composite(panel(bw, hh, round(10 * s), (44, 44, 44), (30, 30, 30)), (margin, margin))
    img.alpha_composite(panel(bw, rh, round(10 * s), (24, 24, 24), (14, 14, 14), border=(55, 55, 55)),
                        (margin, margin + hh + rg))

    # encabezado: avión + subrayado + DEPARTURES
    white = (240, 240, 236, 255)
    icon = plane_icon(round(96 * s))
    ix, iy = margin + round(60 * s), margin + round(hh * 0.44 - icon.height * 0.5)
    layer = Image.new("RGBA", icon.size, white[:3] + (0,))
    layer.putalpha(icon)
    img.alpha_composite(layer, (ix, iy))
    d = ImageDraw.Draw(img)
    uy = margin + round(hh * 0.80)
    d.rectangle((ix - round(4 * s), uy, ix + icon.width + round(4 * s), uy + max(2, round(4 * s))), fill=white)

    font = ImageFont.truetype(FONT_HEAD, round(92 * s))
    tracking = 6 * s
    x = ix + icon.width + round(70 * s)
    cap = font.getbbox("D")
    y = margin + hh / 2 - (cap[1] + cap[3]) / 2
    for ch in HEADER_TEXT:
        d.text((x, y), ch, font=font, fill=white)
        x += d.textlength(ch, font=font) + tracking

    # posiciones de celdas
    cells = []
    x = margin + round(PAD * s)
    y = margin + hh + rg + round(PAD * s)
    for i in range(NAME_CELLS + CODE_CELLS):
        if i == NAME_CELLS:
            x += round((SPACER - GAP) * s)
        cells.append((round(x), y))
        x += (CW + GAP) * s
    return img, cells


# ----------------------------------------------------------------------------
# Animación
# ----------------------------------------------------------------------------
def row_text(name, code):
    return list(name.ljust(NAME_CELLS)[:NAME_CELLS] + code.ljust(CODE_CELLS)[:CODE_CELLS])


def build_timeline():
    """Para cada transición: por celda, (inicio, secuencia de caracteres)."""
    rng = random.Random(7)
    n = NAME_CELLS + CODE_CELLS
    current = [" "] * n
    transitions = []
    t = START
    styles = ["wave_ltr", "random", "cascade_rtl"]
    for (name, code), hold, style in zip(DESTINOS, HOLDS, styles):
        target = row_text(name, code)
        order = list(range(n))
        if style == "random":
            rng.shuffle(order)
            delay = {c: k * 0.018 for k, c in enumerate(order)}
        elif style == "cascade_rtl":
            delay = {c: (n - 1 - c) * 0.028 for c in order}
        else:
            delay = {c: c * 0.022 for c in order}
        cells, end = [], t
        for c in range(n):
            a, b = current[c], target[c]
            if a == b:
                cells.append((t, [a]))
                continue
            steps = rng.randint(3, 7) if style != "cascade_rtl" else rng.randint(5, 9)
            bi = DRUM.index(b)
            seq = [a] + [DRUM[(bi - j) % len(DRUM)] for j in range(steps, 0, -1)] + [b]
            seq = [ch for k, ch in enumerate(seq) if k == 0 or ch != seq[k - 1]]
            start = t + delay[c]
            cells.append((start, seq))
            end = max(end, start + FLIP * (len(seq) - 1))
        transitions.append(cells)
        current = target
        t = end + hold
    return transitions, t


def cell_state(transitions, c, t):
    state = (" ", None, None)
    for cells in transitions:
        start, seq = cells[c]
        if t < start:
            break
        k = (t - start) / FLIP
        i = int(k)
        if i >= len(seq) - 1:
            state = (seq[-1], None, None)
        else:
            state = (seq[i], seq[i + 1], k - i)
            break
    return state


def render(out_dir, fmt):
    W, H, rel = FORMATS[fmt]
    s = W * rel / BOARD_W
    flaps = Flaps(s)
    base, cells = static_board(s, flaps)
    transitions, total = build_timeline()

    if fmt == "alpha":  # solo el cartel, recortado a su tamaño (dimensiones pares)
        W, H = base.width + base.width % 2, base.height + base.height % 2
        bg = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    else:  # fondo de marca con viñeta suave
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((xs - W / 2) / W) ** 2 + ((ys - H / 2) / H) ** 2)
        k = (1.12 - 0.55 * r ** 1.6)[..., None]
        arr = (np.array(BG, np.float32) * k).clip(0, 255).astype(np.uint8)
        bg = Image.fromarray(arr, "RGB").convert("RGBA")
    ox = round((W - base.width) / 2)
    oy = round((H - base.height) / 2)

    frames_dir = Path(out_dir) / f"frames_cartel_{fmt}"
    frames_dir.mkdir(parents=True, exist_ok=True)
    n = math.ceil(total * FPS)
    cache = {}
    for f in range(n):
        t = f / FPS
        frame = bg.copy()
        board = base.copy()
        for c, (x, y) in enumerate(cells):
            a, b, p = cell_state(transitions, c, t)
            if p is None:
                img = cache.get(a)
                if img is None:
                    img = cache[a] = flaps.cell(a, None, None)
            else:
                img = flaps.cell(a, b, p)
            board.alpha_composite(img, (x, y))
        frame.alpha_composite(board, (ox, oy))
        frame.save(frames_dir / f"{f:04d}.png")

    src = ["ffmpeg", "-y", "-v", "error", "-framerate", str(FPS), "-i", str(frames_dir / "%04d.png")]
    if fmt == "alpha":
        # .mov "Animation" (qtrle) con alfa: Premiere, After Effects, Final Cut, DaVinci.
        out = Path(out_dir) / "cartel_destinos_transparente.mov"
        subprocess.run(src + ["-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", "0", "-crf", "24",
                              str(Path(out_dir) / "cartel_destinos_transparente.webm")], check=True)
        cmd = src + ["-c:v", "qtrle", str(out)]
    else:
        out = Path(out_dir) / f"cartel_destinos_{fmt}.mp4"
        cmd = src + ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "slow",
                     "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)
    return out, total


if __name__ == "__main__":
    out_dir = sys.argv[1]
    for fmt in sys.argv[2:] or ["horizontal"]:
        print(*render(out_dir, fmt))
