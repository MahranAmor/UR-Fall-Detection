"""Download the UR Fall Detection dataset (URFD), camera 0 only.

Usage: python scripts/download_urfd.py [--no-rgb]

Gets, into data/raw/urfd/:
  videos/   70 preview mp4s (depth | RGB side by side, 640x240)
  rgb/      cam0 RGB frames, 640x480 PNG, one folder per sequence (~4.3 GB)
  sync/     per-sequence sync csv (frame, ms, interpolated accelerometer)
  labels/   urfall-cam0-falls.csv / -adls.csv (frame labels + depth features)

Re-running skips files that are already there.
Source: http://fenix.ur.edu.pl/~mkepski/ds/uf.html (CC BY-NC-SA 4.0).
"""
import shutil
import ssl
import sys
import urllib.request
import zipfile
from pathlib import Path

# The site redirects to https; some Windows setups reject its cert unless certifi's bundle is used.
try:
    import certifi
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CTX = ssl.create_default_context()

BASE = "http://fenix.ur.edu.pl/~mkepski/ds/data/"
ROOT = Path(__file__).resolve().parent.parent / "data" / "raw" / "urfd"
SEQS = [f"fall-{i:02d}" for i in range(1, 31)] + [f"adl-{i:02d}" for i in range(1, 41)]


def fetch(name, dest):
    if dest.exists() and dest.stat().st_size > 0:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(BASE + name, context=SSL_CTX, timeout=60) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    tmp.replace(dest)
    return True


def main():
    rgb = "--no-rgb" not in sys.argv
    for name in ["urfall-cam0-falls.csv", "urfall-cam0-adls.csv"]:
        fetch(name, ROOT / "labels" / name)
    for i, s in enumerate(SEQS, 1):
        fetch(f"{s}-cam0.mp4", ROOT / "videos" / f"{s}-cam0.mp4")
        fetch(f"{s}-data.csv", ROOT / "sync" / f"{s}-data.csv")
        if rgb:
            out = ROOT / "rgb" / s
            if not (out.exists() and any(out.rglob("*.png"))):
                z = ROOT / "_zips" / f"{s}-cam0-rgb.zip"
                fetch(z.name, z)
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(out)
                z.unlink()
        print(f"[{i}/{len(SEQS)}] {s} done", flush=True)
    print("All done:", ROOT)


if __name__ == "__main__":
    main()
