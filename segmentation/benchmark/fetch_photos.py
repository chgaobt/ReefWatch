"""Download the 100 benchmark photos from CoralNet.

The photos belong to their CoralNet sources, so this repository ships only our masks
and the photo ids. This script fetches each original from its public CoralNet image
page and checks it against the SHA-256 recorded in benchmark.json, so the masks line
up pixel for pixel.

    python benchmark/fetch_photos.py              # all 100, skips ones already present
    python benchmark/fetch_photos.py --limit 3    # quick test

Photos land in benchmark/photos/<location>/<image_id><ext> (git-ignored).
Requests are spaced out (one every 0.5 s by default) to be polite to CoralNet.
"""
import argparse
import hashlib
import html
import json
import re
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urljoin

HERE = Path(__file__).resolve().parent
BASE = "https://coralnet.ucsd.edu"
UA = {"User-Agent": "ReefWatch-benchmark/1.0 (research; coral bleaching segmentation)"}
_last = [0.0]


def fetch(url, interval):
    for attempt in range(3):
        wait = _last[0] + interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.monotonic()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90) as r:
                return r.read()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 + 2 * attempt)


def original_url(page_html):
    """The full-resolution image is the <img> inside #original_image_container."""
    m = re.search(r'id="original_image_container"[\s\S]*?<img[^>]*?\ssrc="([^"]+)"', page_html)
    if not m:
        raise ValueError("original image not found on page (layout changed or image not public)")
    return urljoin(BASE, html.unescape(m.group(1)))


def photo_path(e):
    ext = Path(e.get("original_name") or "x.jpg").suffix.lower() or ".jpg"
    return HERE / "photos" / e["location"] / f"{e['image_id']}{ext}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=0, help="only the first N photos")
    ap.add_argument("--interval", type=float, default=0.5, help="seconds between requests")
    args = ap.parse_args()
    photos = json.loads((HERE / "benchmark.json").read_text(encoding="utf-8"))["photos"]
    if args.limit:
        photos = photos[:args.limit]
    ok = bad = 0
    for i, e in enumerate(photos, 1):
        dest = photo_path(e)
        if dest.is_file() and hashlib.sha256(dest.read_bytes()).hexdigest() == e["sha256"]:
            ok += 1
            continue
        try:
            page = fetch(e["page_url"], args.interval).decode("utf-8", "replace")
            data = fetch(original_url(page), args.interval)
        except Exception as ex:
            print(f"[{i}/{len(photos)}] {e['location']}/{e['image_id']}: FAILED ({ex})", flush=True)
            bad += 1
            continue
        digest = hashlib.sha256(data).hexdigest()
        if digest != e["sha256"]:
            print(f"[{i}/{len(photos)}] {e['location']}/{e['image_id']}: checksum mismatch "
                  f"(CoralNet served a different file); not saved", flush=True)
            bad += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        ok += 1
        print(f"[{i}/{len(photos)}] {e['location']}/{e['image_id']}: ok ({len(data) / 1e6:.1f} MB)", flush=True)
    print(f"done: {ok} ok, {bad} failed")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
