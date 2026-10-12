"""Shared helpers for the handwritten-digit case study.

Stage 1 (acquire) and stage 2 (parse) of Fry's visualization pipeline live here:
the UCI archive is read in place, the 64-feature tables are parsed, and the original
32 x 32 bitmaps are decoded from the Unix-compress (.Z) files without any external tool.
"""
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "data" / "optdigits.zip"
FIG = ROOT / "figures"
RES = ROOT / "results"
DASH = ROOT / "dashboard"
for _d in (FIG, RES, DASH):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 7
PIXELS = [f"p{r}{c}" for r in range(8) for c in range(8)]   # p<row><col>, row 0 = top

# ---------------------------------------------------------------- style
GREEN, PINK, INK, INK2, MUTED, RULE = "#008300", "#e87ba4", "#111111", "#4d4d4d", "#8c8c8c", "#dcdcdc"
GREENS = LinearSegmentedColormap.from_list("greens", ["#f4faf4", "#bfe0bf", "#6fb86f", "#2e8b2e", "#0b4f0b"])
PIYG = LinearSegmentedColormap.from_list("pink_green", ["#b0336a", "#e87ba4", "#f6d3e1", "#f2f2f2",
                                                        "#cce8cc", "#4fa64f", "#005c00"])


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Calibri", "Carlito", "DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "normal", "axes.titlelocation": "left",
        "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "legend.frameon": False, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED,
        "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.7, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": RULE, "grid.linewidth": 0.5,
        "grid.linestyle": "-", "axes.axisbelow": True, "lines.linewidth": 2.0,
        "figure.dpi": 110, "savefig.dpi": 220, "mathtext.fontset": "dejavusans", "axes.unicode_minus": False,
    })


def plain_log(ax, axis="y"):
    """Log-scale ticks written as plain decimals (0.001, 0.01, ...) instead of powers of ten."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    fmt = FuncFormatter(lambda v, _: f"{v:,.6f}".rstrip("0").rstrip(".") if v < 1 else f"{v:,.0f}")
    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_locator(LogLocator(base=10))
    target.set_major_formatter(fmt)
    target.set_minor_formatter(NullFormatter())


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", facecolor="white")
    if "ipykernel" in sys.modules:
        plt.show()
    plt.close(fig)


def dump(name, obj):
    """Merge `obj` into results/<name>.json so every number in the report has a source."""
    path = RES / f"{name}.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    old.update(obj)
    path.write_text(json.dumps(old, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                    encoding="utf-8")


# ---------------------------------------------------------------- acquire / parse
def load_tables():
    """The two preprocessed 8 x 8 tables: 64 integer counts (0-16) + class label."""
    with zipfile.ZipFile(ZIP) as z:
        tra = pd.read_csv(io.BytesIO(z.read("optdigits.tra")), header=None, names=PIXELS + ["digit"])
        tes = pd.read_csv(io.BytesIO(z.read("optdigits.tes")), header=None, names=PIXELS + ["digit"])
    return tra, tes


def unlzw(data: bytes) -> bytes:
    """Decoder for the Unix `compress` (.Z, LZW) format used by the original UCI bitmap files."""
    if data[:2] != b"\x1f\x9d":
        raise ValueError("not a .Z file")
    maxbits, block_mode = data[2] & 0x1F, bool(data[2] & 0x80)
    out = bytearray()
    table = [bytes([i]) for i in range(256)]
    if block_mode:
        table.append(b"")                         # code 256 = CLEAR
    n_bits, pos = 9, 3 * 8                        # bit position after the 3-byte header
    group_start = pos
    prev = None
    total_bits = len(data) * 8
    while pos + n_bits <= total_bits:
        byte_i, bit_i = divmod(pos, 8)
        chunk = int.from_bytes(data[byte_i:byte_i + 4].ljust(4, b"\0"), "little")
        code = (chunk >> bit_i) & ((1 << n_bits) - 1)
        pos += n_bits
        if block_mode and code == 256:
            # codes are written in groups of n_bits bytes; skip to the end of the current group
            used = pos - group_start
            pos += (-used) % (n_bits * 8)
            table = table[:257]
            n_bits, prev = 9, None
            group_start = pos
            continue
        if code < len(table):
            entry = table[code]
            if prev is not None:
                table.append(prev + entry[:1])
        elif code == len(table) and prev is not None:
            entry = prev + prev[:1]
            table.append(entry)
        else:
            raise ValueError("corrupt LZW stream")
        out += entry
        prev = entry
        if len(table) >= (1 << n_bits) and n_bits < maxbits:
            used = pos - group_start
            pos += (-used) % (n_bits * 8)
            n_bits += 1
            group_start = pos
    return bytes(out)


def load_bitmaps(member="optdigits-orig.windep.Z"):
    """Original 32 x 32 binary bitmaps and labels from one of the .Z files."""
    with zipfile.ZipFile(ZIP) as z:
        text = unlzw(z.read(member)).decode("ascii")
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if len(ln.strip()) == 32 and set(ln.strip()) <= {"0", "1"})
    imgs, labels, i = [], [], start
    while i + 32 < len(lines):
        block = lines[i:i + 32]
        if not all(len(b.strip()) == 32 for b in block):
            break
        imgs.append(np.array([[int(ch) for ch in b.strip()] for b in block], dtype=np.uint8))
        labels.append(int(lines[i + 32].strip()))
        i += 33
    return np.stack(imgs), np.array(labels)


def to_images(df):
    return df[PIXELS].to_numpy(np.float32).reshape(-1, 8, 8)
