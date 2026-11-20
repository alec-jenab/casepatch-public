"""Portable font lookup for rendering replacement text.

Order: fontconfig (`fc-match`, Linux and Homebrew setups) -> a scan of the platform's font folders -> the fonts bundled
in casepatch/fonts (Liberation Sans/Serif/Mono, DejaVu Sans). Folders:
  Linux:   /usr/share/fonts, /usr/local/share/fonts, ~/.local/share/fonts, ~/.fonts
  macOS:   /System/Library/Fonts (+ Supplemental), /Library/Fonts, ~/Library/Fonts
  Windows: %WINDIR%\\Fonts, %LOCALAPPDATA%\\Microsoft\\Windows\\Fonts
plus CASEPATCH_FONT_DIRS (os.pathsep-separated) first. Nothing is downloaded.
"""
from __future__ import annotations

import functools
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

BUNDLED = Path(__file__).resolve().parents[1] / "fonts"
FONT_EXT = {".ttf", ".otf", ".ttc"}
# metric-compatible stand-ins, best first (Liberation = Arimo/Tinos/Cousine metrics = Arial/Times/Courier)
ALIASES = {
    "sans": ["Liberation Sans", "Arimo", "Arial", "Helvetica", "Helvetica Neue", "DejaVu Sans", "Verdana"],
    "serif": ["Liberation Serif", "Tinos", "Times New Roman", "Times", "DejaVu Serif", "Georgia"],
    "mono": ["DejaVu Sans Mono", "Liberation Mono", "Cousine", "Courier New", "Courier", "Menlo", "Consolas"],
}
BUNDLED_FILES = {("sans", False): "LiberationSans-Regular.ttf", ("sans", True): "LiberationSans-Bold.ttf",
                 ("serif", False): "LiberationSerif-Regular.ttf", ("serif", True): "LiberationSerif-Bold.ttf",
                 ("mono", False): "LiberationMono-Regular.ttf", ("mono", True): "LiberationMono-Bold.ttf"}


def font_dirs(platform: str | None = None, env: dict | None = None, home: Path | None = None) -> list[Path]:
    platform = platform or sys.platform
    env = os.environ if env is None else env
    home = Path(home) if home is not None else Path.home()
    dirs = [Path(p) for p in (env.get("CASEPATCH_FONT_DIRS") or "").split(os.pathsep) if p]
    if platform == "darwin":
        dirs += [Path("/System/Library/Fonts"), Path("/System/Library/Fonts/Supplemental"), Path("/Library/Fonts"),
                 home / "Library" / "Fonts"]
    elif platform.startswith("win"):
        dirs += [Path(env.get("WINDIR") or env.get("SystemRoot") or r"C:\Windows") / "Fonts"]
        if env.get("LOCALAPPDATA"):
            dirs.append(Path(env["LOCALAPPDATA"]) / "Microsoft" / "Windows" / "Fonts")
    else:
        dirs += [Path("/usr/share/fonts"), Path("/usr/local/share/fonts"), home / ".local" / "share" / "fonts", home / ".fonts"]
    return dirs + [BUNDLED]


def _style_of(name: str) -> tuple[bool, bool]:
    s = name.lower()
    return ("bold" in s or "black" in s or "heavy" in s or "semibold" in s), ("italic" in s or "oblique" in s)


def _names(path: Path):
    """(family, style) from the font's name table; filename heuristics when the font can't be opened."""
    try:
        from PIL import ImageFont
        fam, style = ImageFont.truetype(str(path), 12).getname()
        return fam or path.stem, style or "Regular"
    except Exception:
        stem = path.stem
        fam, _, style = stem.partition("-")
        return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", fam), style or "Regular"


@functools.lru_cache(maxsize=4)
def _index(dirs: tuple[str, ...]) -> dict:
    """family (lowercase) -> {(bold, italic): path}; first folder wins."""
    idx: dict = {}
    for d in dirs:
        p = Path(d)
        if not p.is_dir():
            continue
        try:
            files = sorted(f for f in p.rglob("*") if f.suffix.lower() in FONT_EXT and f.is_file())
        except OSError:
            continue
        for f in files[:5000]:
            fam, style = _names(f)
            key = _style_of(style)
            idx.setdefault(fam.strip().lower(), {}).setdefault(key, str(f))
    return idx


def index(dirs=None) -> dict:
    return _index(tuple(str(d) for d in (dirs if dirs is not None else font_dirs())))


def _fc_match(family: str, style: str):
    if not shutil.which("fc-match"):
        return None
    try:
        out = subprocess.run(["fc-match", "-f", "%{family[0]}|%{file}", f"{family}:style={style}"], capture_output=True,
                             text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    fam, _, path = out.partition("|")
    return path if fam.strip().lower() == family.lower() and path and os.path.exists(path) else None


@functools.lru_cache(maxsize=None)
def find(family: str, style: str = "Regular") -> str | None:
    """Path of `family` in `style` (Regular/Bold/Italic/Bold Italic), exact family only; None when not installed."""
    p = _fc_match(family, style)
    if p:
        return p
    faces = index().get(family.strip().lower())
    if not faces:
        return None
    want = _style_of(style)
    return faces.get(want) or (faces.get((want[0], False)) if want[1] else None) or (faces.get((False, False)) if want == (False, False) else None)


def resolve(cls: str, bold: bool = False) -> str:
    """Best available face for a class ('sans'/'serif'/'mono'): metric-compatible system fonts, else bundled Liberation."""
    style = "Bold" if bold else "Regular"
    for fam in ALIASES.get(cls, ALIASES["sans"]):
        p = find(fam, style)
        if p:
            return p
    return str(BUNDLED / BUNDLED_FILES[(cls if cls in ALIASES else "sans", bold)])


def fallback_any() -> list[str]:
    """Last-resort faces when nothing else is found (bundled, always present)."""
    return [str(BUNDLED / "LiberationSans-Regular.ttf"), str(BUNDLED / "DejaVuSans.ttf"), str(BUNDLED / "LiberationMono-Regular.ttf")]
