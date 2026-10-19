"""Score a checkpoint on the ReefWatch bleaching benchmark (100 held-out photos).

    python benchmark/fetch_photos.py                                   # once
    python eval/evaluate.py weights/MILESTONE_R34_DAPPM_7c.pt          # writes results next to the checkpoint

Reports, per location (Kenya, Red Sea, Chagos, Norfolk Island, La Paz) and pooled:
  * bleaching: BLEACHED IoU, precision, recall, F1 and the binary "bleached vs not
    bleached" mIoU = (IoU_bleached + IoU_not_bleached) / 2  -- the headline number;
  * mIoU under three class schemes: 7-class, 6-class lenient (DEAD and ROCK merged,
    since dead-coral-vs-rock is ambiguous even for people) and 4-class health
    (LIVE / BLEACHED / DEGRADED = DEAD+ROCK / OTHER = SAND+OCEAN+OTHER).

"Pooled" = one confusion matrix summed over all 100 photos (pixels, IGNORE excluded),
the same way CoralScapes reports mIoU. Inference: each photo is resized so its long side
is 1024 px, centre-padded to a multiple of 256, predicted in one pass, and the padding
cropped off -- exactly how the model was validated.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmark"
sys.path.insert(0, str(ROOT / "train"))

IGNORE = 255
SEVEN = ["BLEACHED", "DEAD", "ALIVE", "ROCK", "OCEAN", "SAND", "OTHER"]   # model output order
LOCS = ["kenya", "red_sea", "chagos", "norfolk", "la_paz"]
#: 7-class index -> scheme index, and the scheme's class names
SCHEMES = {
    "7c": ([0, 1, 2, 3, 4, 5, 6], SEVEN),
    "6c_lenient": ([0, 1, 2, 1, 3, 4, 5], ["BLEACHED", "DEAD_ROCK", "ALIVE", "OCEAN", "SAND", "OTHER"]),
    "4c_health": ([1, 2, 0, 2, 3, 3, 3], ["LIVE", "BLEACHED", "DEGRADED", "OTHER"]),
}


def map_ids(a, lut):
    t = np.full(256, IGNORE, np.uint8)
    t[:len(lut)] = lut
    return t[a]


def resize_pad(img, mask, long_side=1024, multiple=256):
    """Long side -> 1024 (bilinear image, nearest mask), centre-pad the IMAGE to a multiple of 256.
    Returns (padded image, resized mask WITHOUT padding, (top, left, nh, nw))."""
    h, w = img.shape[:2]
    s = float(long_side) / max(w, h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    im = np.asarray(Image.fromarray(img).resize((nw, nh), Image.BILINEAR), np.uint8)
    m = np.asarray(Image.fromarray(mask, mode="L").resize((nw, nh), Image.NEAREST), np.uint8)
    tw, th = -(-nw // multiple) * multiple, -(-nh // multiple) * multiple
    left, top = (tw - nw) // 2, (th - nh) // 2
    out = np.empty((th, tw, 3), np.uint8)
    out[:] = np.array([124, 116, 104], np.uint8)          # ImageNet mean pixel
    out[top:top + nh, left:left + nw] = im
    return out, m, (top, left, nh, nw)


def load_benchmark():
    spec = json.loads((BENCH / "benchmark.json").read_text(encoding="utf-8"))
    items = []
    for e in spec["photos"]:
        photos = sorted((BENCH / "photos" / e["location"]).glob(f"{e['image_id']}.*"))
        if not photos:
            raise SystemExit(f"missing photo {e['location']}/{e['image_id']}: run benchmark/fetch_photos.py first")
        with Image.open(photos[0]) as im:
            img = np.asarray(im.convert("RGB"), np.uint8)
        m = np.asarray(Image.open(BENCH / e["mask"]))                       # export ids 1..7, 255
        gt = np.where((m >= 1) & (m <= 7), m - 1, IGNORE).astype(np.uint8)  # -> 7-class index
        gt = np.asarray(Image.fromarray(gt, mode="L").resize((img.shape[1], img.shape[0]), Image.NEAREST))
        pad, gt_small, box = resize_pad(img, gt)        # gt_small is nh x nw: the unpadded photo area
        items.append(dict(loc=e["location"], id=e["image_id"], img=pad, box=box, gt=gt_small))
    return items


def confusion(gt, pred, n):
    keep = (gt != IGNORE) & (pred < n)
    return np.bincount(gt[keep].astype(np.int64) * n + pred[keep], minlength=n * n).reshape(n, n)


def miou(cm):
    tp = np.diag(cm).astype(np.float64)
    union = cm.sum(0) + cm.sum(1) - tp
    iou = np.where(union > 0, tp / np.maximum(union, 1), np.nan)
    return float(np.nanmean(iou)), iou


def bleach_scores(tp, fp, fn, tn):
    iou = tp / (tp + fp + fn)
    iou_rest = tn / (tn + fp + fn)
    return dict(bleached_iou=iou, not_bleached_iou=iou_rest, binary_miou=(iou + iou_rest) / 2,
                precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1), f1=2 * tp / (2 * tp + fp + fn),
                gt_bleached_share=(tp + fn) / (tp + fp + fn + tn))


def main():
    ap = argparse.ArgumentParser(description="Score a checkpoint on the 100-photo benchmark")
    ap.add_argument("checkpoint")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="", help="results json (default: <checkpoint>.benchmark.json)")
    args = ap.parse_args()
    import standalone_train as S

    items = load_benchmark()
    print(f"benchmark: {len(items)} photos", flush=True)
    dev = torch.device(args.device)
    model, ck = S.load_ckpt(args.checkpoint, dev)
    nc = int(ck.get("num_classes", 4))
    if nc not in (4, 7):
        raise SystemExit(f"checkpoint has {nc} classes; only 7-class and 4-class health models are scored")
    # A 4-class health model (LIVE, BLEACHED, DEGRADED, OTHER) is scored on bleaching and 4c_health only.
    bleached_id = 0 if nc == 7 else 1
    schemes = SCHEMES if nc == 7 else {"4c_health": SCHEMES["4c_health"]}
    predict = S.make_predictor(model, dev)
    t0 = time.time()
    preds = []
    for it in items:
        t, l, nh, nw = it["box"]
        preds.append(predict(it["img"]).cpu().numpy().astype(np.uint8)[t:t + nh, l:l + nw])
    secs = time.time() - t0

    res = {"checkpoint": Path(args.checkpoint).name, "num_classes": nc, "n_photos": len(items), "bleaching": {}, "miou": {}}
    for scope in LOCS + ["pooled"]:
        sel = [i for i, it in enumerate(items) if scope == "pooled" or it["loc"] == scope]
        c = np.zeros(4, np.int64)                                       # tp, fp, fn, tn for BLEACHED
        for i in sel:
            g, p = items[i]["gt"], preds[i]
            valid = g != IGNORE
            gb, pb = g == 0, p == bleached_id                           # GT index 0 = BLEACHED
            c += [np.sum(gb & pb & valid), np.sum(~gb & pb & valid), np.sum(gb & ~pb & valid), np.sum(~gb & ~pb & valid)]
        res["bleaching"][scope] = {k: round(float(v), 4) for k, v in bleach_scores(*[float(x) for x in c]).items()}
        for name, (lut, names) in schemes.items():
            pr = (lambda a: map_ids(a, lut)) if nc == 7 else (lambda a: a)   # 4-class output is already health ids
            cm = sum(confusion(map_ids(items[i]["gt"], lut), pr(preds[i]), len(names)) for i in sel)
            m, iou = miou(cm)
            res["miou"].setdefault(name, {})[scope] = {
                "miou": round(m, 4), "per_class_iou": {n: (None if np.isnan(v) else round(float(v), 4)) for n, v in zip(names, iou)}}

    b = res["bleaching"]["pooled"]
    print(f"\nBLEACHING (pooled): binary mIoU {b['binary_miou']:.3f} | bleached IoU {b['bleached_iou']:.3f} | "
          f"P {b['precision']:.3f} R {b['recall']:.3f} F1 {b['f1']:.3f}")
    print("bleached IoU per location: " + "  ".join(f"{l} {res['bleaching'][l]['bleached_iou']:.3f}" for l in LOCS))
    for name in schemes:
        print(f"mIoU {name:<11} pooled {res['miou'][name]['pooled']['miou']:.3f} | " +
              "  ".join(f"{l} {res['miou'][name][l]['miou']:.3f}" for l in LOCS))
    print(f"\n{len(items)} photos in {secs:.1f} s on {dev}")
    out = Path(args.out) if args.out else Path(args.checkpoint).with_suffix(".benchmark.json")
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
