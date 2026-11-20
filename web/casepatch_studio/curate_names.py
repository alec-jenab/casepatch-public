"""Build names.json: the replacement-name pool Studio uses, keeping only names whose glyphs survive a photo-like restyle.

Each candidate is drawn the way a redrawn ID field looks (small, bold, soft ink on a pale card, slight rotation, blur, JPEG)
in several sans fonts and read back with CasePatch's OCR engine. A name is kept only if every variant reads back exactly.
Names are also screened for shapes that smear at small sizes (m, w, rn, cl, double letters such as 'll' read as 'II').
Usage: python web/casepatch_studio/curate_names.py   (writes names.json next to this file; deterministic; local only)
"""
from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
         "/usr/share/fonts/truetype/sand-box/google/Fira Sans/FiraSans-SemiBold.ttf",
         "/usr/share/fonts/truetype/sand-box/google/Carlito/Carlito-Regular.ttf",
         "/usr/share/fonts/truetype/sand-box/google/Work Sans/WorkSans-VariableFont_wght.ttf"]
# feminine-, masculine-styled candidates per initial (the engine's pool alternates f/m by index)
CANDIDATES = {
    "A": (["Alma", "Ada", "Anna", "Aria", "Alice"], ["Anton", "Arlo", "Abel", "Axel", "Asher"]),
    "B": (["Bea", "Bianca", "Belle", "Brynn", "Beth"], ["Boris", "Bram", "Basil", "Brody", "Byron"]),
    "C": (["Clara", "Celia", "Cora", "Carla", "Chloe"], ["Cyrus", "Carl", "Calvin", "Cedric", "Colin"]),
    "D": (["Dana", "Delia", "Daria", "Dora", "Diana"], ["Dax", "Dirk", "David", "Dean", "Doran"]),
    "E": (["Elena", "Elsa", "Eva", "Edith", "Erika"], ["Ezra", "Emil", "Elias", "Ethan", "Edgar"]),
    "F": (["Freya", "Fern", "Fiona", "Flora", "Faye"], ["Felix", "Flynn", "Frank", "Fabian", "Ford"]),
    "G": (["Greta", "Gwen", "Grace", "Gina", "Gaia"], ["Gideon", "Gus", "Grant", "Glen", "Gavin"]),
    "H": (["Hazel", "Hana", "Helen", "Heidi", "Hope"], ["Hugo", "Hal", "Henry", "Hector", "Harris"]),
    "I": (["Ingrid", "Isla", "Iris", "Ida", "Irene"], ["Ivo", "Ike", "Ivan", "Isaac", "Igor"]),
    "J": (["Juno", "Jolene", "Julia", "Jane", "Josie"], ["Jasper", "Joel", "Jonas", "Jude", "Jake"]),
    "K": (["Kira", "Kaia", "Katya", "Kate", "Kelsey"], ["Kurt", "Knox", "Karl", "Kevin", "Kai"]),
    "L": (["Lydia", "Lara", "Lena", "Lucia", "Laura"], ["Leon", "Lev", "Lucas", "Lars", "Logan"]),
    "M": (["Maren", "Mabel", "Maya", "Nora", "Marta"], ["Milo", "Moss", "Marco", "Mason", "Miles"]),
    "N": (["Nadia", "Nell", "Nora", "Nina", "Naya"], ["Nolan", "Niko", "Nash", "Noah", "Neil"]),
    "O": (["Odette", "Opal", "Olga", "Olive", "Ora"], ["Otto", "Orrin", "Oscar", "Owen", "Oren"]),
    "P": (["Petra", "Pia", "Paula", "Paige", "Priya"], ["Pavel", "Pike", "Peter", "Paul", "Pablo"]),
    "Q": (["Quinn", "Quilla", "Queenie", "Qiana", "Quella"], ["Quentin", "Quade", "Quincy", "Quill", "Quan"]),
    "R": (["Rosa", "Rhea", "Ruth", "Rita", "Riley"], ["Ronan", "Rafe", "Ralph", "Ross", "Rex"]),
    "S": (["Sage", "Selma", "Sara", "Stella", "Sofia"], ["Silas", "Soren", "Seth", "Simon", "Stefan"]),
    "T": (["Tess", "Talia", "Tara", "Tina", "Thea"], ["Tobias", "Theo", "Tristan", "Troy", "Todd"]),
    "U": (["Uma", "Una", "Ursula", "Ulla", "Uta"], ["Uri", "Ulric", "Ugo", "Upton", "Ulf"]),
    "V": (["Vera", "Vida", "Viola", "Vesna", "Vivian"], ["Viggo", "Vaughn", "Victor", "Vince", "Vadim"]),
    "W": (["Wren", "Willa", "Wanda", "Winona", "Wendy"], ["Wade", "Wolf", "Walter", "Wes", "Wilbur"]),
    "X": (["Xenia", "Xara", "Ximena", "Xiomara", "Xena"], ["Xavi", "Xan", "Xander", "Xavier", "Xerxes"]),
    "Y": (["Yara", "Yvette", "Yasmin", "Yolanda", "Yuki"], ["Yusuf", "Yann", "Yuri", "Yosef", "Yves"]),
    "Z": (["Zara", "Zelda", "Zoe", "Zita", "Zora"], ["Zane", "Zev", "Zack", "Zeke", "Zoran"]),
}
SURNAMES = ["Roe", "Fox", "Ley", "Kim", "Orr", "Poe", "Ash", "Bell", "Cole", "Dunn", "Hale", "Lund", "Moss", "Pratt", "Brooks",
            "Calloway", "Draper", "Ellison", "Fairbanks", "Garrow", "Holloway", "Ingram", "Jessup", "Kessler", "Lindqvist",
            "Marchetti", "Northcott", "Okafor", "Pemberton", "Quarles", "Rasmussen", "Sorensen", "Thackeray", "Underhill",
            "Vasquez", "Whitlock", "Yardley", "Zielinski", "Ashdown", "Birch", "Corrigan", "Delacroix", "Carver", "Doyle",
            "Easton", "Fisher", "Grant", "Harper", "Keane", "Larsen", "Parker", "Reyes", "Sutton", "Tate", "Vance", "Porter",
            "Hayes", "Dalton", "Becker", "Stone", "Ford", "Lane", "Price", "Reid", "Shaw", "Cruz", "Diaz", "Boyd", "Kerr"]
RISKY = re.compile(r"[mw]|rn|cl|(.)\1")  # smear or merge at ~25 px after blur (the 'Esme' -> 'Esnie' failure)


def variants(text):
    for fp in FONTS:
        if not Path(fp).exists():
            continue
        for size, blur, angle in ((26, 1.4, -14), (30, 1.8, 0)):
            f = ImageFont.truetype(fp, size)
            w = max(320, int(f.getlength(text)) + 160)  # OCR text detection needs some surrounding margin
            im = Image.new("RGB", (w, 180), (206, 232, 228))
            ImageDraw.Draw(im).text((80, 70), text, font=f, fill=(92, 100, 96))
            im = im.rotate(angle, expand=True, fillcolor=(206, 232, 228), resample=Image.BICUBIC).filter(ImageFilter.GaussianBlur(blur))
            buf = io.BytesIO(); im.save(buf, "JPEG", quality=80); buf.seek(0)
            yield Image.open(buf).convert("RGB")


def reads_back(text, ocr):
    import numpy as np
    for im in variants(text):
        got = " ".join(l.text for l in ocr(np.asarray(im), "c")).strip()
        if got.replace(" ", "").lower() != text.lower():
            return False
    return True


CACHE = Path.home() / ".cache" / "casepatch" / "name-readback.json"


def main():
    from casepatch.docs.extract import ocr_lines
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    def ok(n):
        if RISKY.search(n):
            return False
        if n not in cache:
            cache[n] = reads_back(n, ocr_lines)
            CACHE.parent.mkdir(parents=True, exist_ok=True); CACHE.write_text(json.dumps(cache))
        return cache[n]
    given, rejected = {}, []
    for letter, (fem, masc) in CANDIDATES.items():
        f = [n for n in fem if n[0] == letter and ok(n) or (rejected.append(n) and False)]
        m = [n for n in masc if n[0] == letter and ok(n) or (rejected.append(n) and False)]
        k = min(len(f), len(m), 3)
        if k == 0:
            raise SystemExit(f"no usable names for {letter}: f={f} m={m}")
        given[letter] = [x for pair in zip(f[:k], m[:k]) for x in pair]
    surnames = [s for s in SURNAMES if ok(s) or (rejected.append(s) and False)]
    out = {"note": "Generated by curate_names.py: OCR readback-checked replacement names for Studio (fonts: "
                   + ", ".join(Path(f).stem for f in FONTS if Path(f).exists()) + ").",
           "given": given, "surnames": surnames, "rejected": sorted(set(rejected))}
    Path(__file__).with_name("names.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"given={sum(map(len, given.values()))} surnames={len(surnames)} rejected={len(set(rejected))}")


if __name__ == "__main__":
    main()
