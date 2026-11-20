"""Model/font assets for the local engine: check (`doctor`) and one-time download (`fetch`).

Downloads only happen when the user runs setup (studio.py does it once, with a printed list) and never during a
case run. Every file is pinned by SHA-256. Licenses: LaMa big-lama.pt (Apache-2.0, via simple-lama-inpainting
v0.1.0), YuNet face detector (MIT, OpenCV Zoo), SFace face recognizer (Apache-2.0, OpenCV Zoo), SFHQ synthetic face
sample grid (MIT on GitHub / CC0 on Kaggle; used only as replacement faces).
"""
from __future__ import annotations

import hashlib
import http.client
import importlib
import os
import shutil
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

ASSETS = {
    "big-lama.pt": ("https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt",
                    "7ba7aa7ac37a4d41fdbbeba3a2af7ead18058552997e3a3cd1a3b2210c9e6b4c", 206, "inpainting under replaced text (falls back to OpenCV inpainting)"),
    "face_detection_yunet_2023mar.onnx": ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
                                          "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4", 0.3, "face detection (faces would not be found)"),
    "face_recognition_sface_2021dec.onnx": ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
                                            "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79", 39, "check that a swapped face is a different person"),
    "sfhq_sample_4x8.jpg": ("https://raw.githubusercontent.com/SelfishGene/SFHQ-dataset/main/images/SFHQ_sample_4x8.jpg",
                            "9c7c9be40199f5628efd0c5baa44a2f856e7a728ac2532aba266dcc2416ed3b8", 2, "synthetic replacement faces"),
}
PACKAGES = ["pymupdf", "PIL", "numpy", "cv2", "rapidocr", "onnxruntime", "torch", "segno", "zxingcpp", "fontTools"]


class DownloadError(RuntimeError):
    """A model file could not be downloaded; the message is one line with the file, host, reason and a fix."""


def _context():
    try:
        from casepatch.tls import ssl_context
    except ImportError:  # a source checkout run without the casepatch package on the path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from casepatch.tls import ssl_context
    return ssl_context()


def _hint(err):
    from casepatch.tls import network_hint
    return network_hint(err)


def model_dir() -> Path:
    try:
        from casepatch.paths import model_dir as _md
    except ImportError:  # a source checkout run without the casepatch package on the path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from casepatch.paths import model_dir as _md
    return _md()


def _find(name):
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    try:
        from casepatch.docs.models import find_model
        return find_model(name)
    except Exception:
        p = model_dir() / name
        return p if p.exists() else None


def doctor() -> dict:
    missing_pkgs = []
    for m in PACKAGES:
        try:
            importlib.import_module(m)
        except Exception:
            missing_pkgs.append(m)
    assets = {n: str(_find(n) or "") for n in ASSETS}
    return {"python": sys.version.split()[0], "missing_packages": missing_pkgs,
            "assets": assets, "missing_assets": [n for n, p in assets.items() if not p],
            "fc_match": bool(shutil.which("fc-match"))}


def fetch(names=None, quiet=False):
    dest = model_dir(); dest.mkdir(parents=True, exist_ok=True)
    for name in names or ASSETS:
        if _find(name):
            continue
        url, sha, mb, _ = ASSETS[name]
        if not quiet:
            print(f"  downloading {name} (~{mb} MB) from {url.split('/')[2]}", flush=True)
        tmp = dest / (name + ".part")
        host = url.split('/')[2]
        try:
            with urllib.request.urlopen(url, timeout=120, context=_context()) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)
        except (urllib.error.URLError, http.client.HTTPException, ssl.SSLError, TimeoutError, ConnectionError) as e:
            tmp.unlink(missing_ok=True)
            reason = f"HTTP {e.code}" if isinstance(e, urllib.error.HTTPError) else str(getattr(e, "reason", None) or e)
            raise DownloadError(" ".join(filter(None, [f"Could not download {name} from {host}: {reason}.", _hint(e)]))) from None
        h = hashlib.sha256(tmp.read_bytes()).hexdigest()
        if h != sha:
            tmp.unlink()
            raise DownloadError(f"{name}: checksum mismatch; not installed")
        tmp.rename(dest / name)


if __name__ == "__main__":
    import json
    if sys.argv[1:] == ["fetch"]:
        fetch()
    print(json.dumps(doctor(), indent=1))
