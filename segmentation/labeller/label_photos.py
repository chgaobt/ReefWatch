r"""Label coral reef photos with 7 classes: SAM regions + an LLM answering per region ("M7C" labeller).

This is the labeller that produced the 7-class training masks for the milestone segmentation model.
For every photo it runs five steps:

  1. Regions.  Resize the photo so its long side is 1280 px (the "key"). Run SAM v1 ViT-B automatic mask
     generation on the key (scaled to 1024 px), paint the masks into a region map largest-first so smaller
     masks stay on top, fill the pixels SAM left uncovered with SLIC-style superpixels, split every region
     into connected components, and merge small regions into their longest-border neighbour until every
     region is >= 0.15 % of the image and there are at most 80. Regions are numbered in reading order.
  2. Overlay.  Draw the region outlines and numbers on the key; this is the image the model reads.
  3. Ask.      Call Claude Sonnet headlessly through the `claude` CLI with the M7C prompt
               (prompt_m7c.txt). The model reads the clean key and the overlay with its Read tool and
               answers JSON: {"regions": {"1": {"type": "hard_coral", "score": 2}, ...}, "notes": ...}.
  4. Parse.    Extract and validate the JSON (at most max(2, 10 %) of the regions may be missing) and map
               each answer to a class: hard_coral with CoralWatch score <= 3 -> BLEACHED, 4-6 -> ALIVE;
               dead_coral -> DEAD, rock -> ROCK, sand -> SAND, water -> OCEAN, other -> OTHER;
               a region with no answer -> IGNORE.
  5. Mask.     Paint the region classes into a mask at key resolution, nearest-upsample it to the original
               photo size and save it as a single-channel PNG, next to the raw JSON answer.

Mask ids: 1 BLEACHED, 2 DEAD, 3 ALIVE, 4 ROCK, 5 OCEAN, 6 SAND, 7 OTHER, 255 IGNORE (0 is never written).
Photos are read as raw pixels (EXIF orientation is NOT applied), so masks align with the stored pixel grid.

    python label_photos.py photos/ --location "Norfolk Island (South Pacific)" \
        --sam-weights sam_vit_b_01ec64.pth --out labels/
    python label_photos.py photo.jpg --location "the Red Sea" --sam-weights sam_vit_b_01ec64.pth \
        --out labels/ --dry-run
"""
import argparse
import hashlib
import io
import json
import random
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
PROMPT_FILE = HERE / 'prompt_m7c.txt'
PROMPT_SHA256 = '942b15270467f620371628c34ca492e44dcb139eb02c8b762e5b803b4ababa1e'   # the template that was used

DEFAULT_MODEL = 'claude-sonnet-5-5'
SYSTEM_PROMPT = 'You label coral reef photos. Use the Read tool to view image files.'
LONG = 1280                 # long side of the key every region map and answer refers to
SAM_SIDE = 1024             # SAM runs on the key scaled to this long side
SAM_SETTINGS = dict(points_per_side=32, pred_iou_thresh=0.80, stability_score_thresh=0.85, min_mask_region_area=200)
MIN_MASK_FRAC = 0.001       # drop SAM masks smaller than this fraction of the image
SLIC_SIDE, SLIC_K = 256, 70  # gap-fill superpixels: computed at 256 px, ~70 seeds
MIN_REGION_FRAC = 0.0015    # merge regions smaller than this fraction of the image
MAX_REGIONS = 80
BLEACHED_MAX_SCORE = 3      # CoralWatch brightness <= 3 counts as bleached

CLASS_IDS = {'BLEACHED': 1, 'DEAD': 2, 'ALIVE': 3, 'ROCK': 4, 'OCEAN': 5, 'SAND': 6, 'OTHER': 7}
IGNORE_ID = 255
CLASS_OF_TYPE = {'dead_coral': 'DEAD', 'rock': 'ROCK', 'sand': 'SAND', 'water': 'OCEAN', 'other': 'OTHER'}
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp'}


# --------------------------------------------------------------------------------------------- step 1
def load_photo(path):
    """Original photo (raw pixels), the 1280-px key, and the key after a quality-90 JPEG round trip.

    The original pipeline stored the key as a q90 JPEG and computed regions and overlays from that file,
    so the round trip is reproduced here; the model's clean copy is written separately at q92."""
    raw = Image.open(path).convert('RGB')
    s = LONG / max(raw.size)
    small = raw.resize((round(raw.width * s), round(raw.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    small.save(buf, format='JPEG', quality=90)
    buf.seek(0)
    key = Image.open(buf).convert('RGB')
    return raw, small, key


def build_sam(weights):
    import torch
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    sam = sam_model_registry['vit_b'](checkpoint=str(weights)).to('cuda' if torch.cuda.is_available() else 'cpu')
    return SamAutomaticMaskGenerator(sam, **SAM_SETTINGS)


def slic_labels(img_arr, k=90, m=18, iters=8):
    """SLIC-style k-means in Lab colour + xy, vectorised. Labels start at 1."""
    import cv2
    h, w = img_arr.shape[:2]
    labc = cv2.cvtColor(img_arr, cv2.COLOR_RGB2LAB).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    S = np.sqrt(h * w / k)
    gy, gx = np.arange(S / 2, h, S), np.arange(S / 2, w, S)
    cy, cx = [a.ravel() for a in np.meshgrid(gy, gx, indexing='ij')]
    feats = np.concatenate([labc.reshape(-1, 3), (m / S) * np.stack([yy.ravel(), xx.ravel()], 1)], 1)
    cent = np.concatenate([labc[cy.astype(int), cx.astype(int)], (m / S) * np.stack([cy, cx], 1)], 1)
    ff = (feats ** 2).sum(1, keepdims=True)
    for _ in range(iters):
        d = ff - 2 * feats @ cent.T + (cent ** 2).sum(1)[None, :]
        a = d.argmin(1)
        cnt = np.bincount(a, minlength=len(cent)).astype(np.float32)
        for j in range(feats.shape[1]):
            sm = np.bincount(a, weights=feats[:, j], minlength=len(cent))
            cent[:, j] = np.where(cnt > 0, sm / np.maximum(cnt, 1), cent[:, j])
    return a.reshape(h, w) + 1


def connected_components(lab):
    """Give every 4-connected piece of every region its own id."""
    from scipy import ndimage
    out = np.zeros_like(lab)
    nxt = 1
    for v in np.unique(lab):
        if v == 0:
            continue
        comp, nc = ndimage.label(lab == v)
        for c in range(1, nc + 1):
            out[comp == c] = nxt
            nxt += 1
    return out


def merge_small(lab, min_area, max_regions):
    """Merge the smallest region into the neighbour it shares the longest border with, until every region
    is >= min_area and there are <= max_regions; renumber 1..N in reading order (8 horizontal bands)."""
    from scipy import ndimage
    lab = lab.copy()
    H, W = lab.shape
    areas = Counter(lab.ravel().tolist())
    areas.pop(0, None)
    while areas:
        n = len(areas)
        target, a = min(areas.items(), key=lambda kv: kv[1])
        if a >= min_area and n <= max_regions:
            break
        ys, xs = np.nonzero(lab == target)
        y0, y1 = max(0, ys.min() - 2), min(H, ys.max() + 3)
        x0, x1 = max(0, xs.min() - 2), min(W, xs.max() + 3)
        sub = lab[y0:y1, x0:x1]
        m = sub == target
        ring = ndimage.binary_dilation(m, iterations=1) & ~m
        vals = Counter(sub[ring].tolist())
        vals.pop(0, None)
        vals.pop(int(target), None)
        if not vals:
            areas.pop(target)
            continue
        dest = vals.most_common(1)[0][0]
        sub[m] = dest
        areas[dest] += areas.pop(target)
    ids = sorted(areas)
    cents = ndimage.center_of_mass(np.ones_like(lab), lab, ids)
    order = sorted(range(len(ids)), key=lambda k: (int(cents[k][0] // (H / 8)), cents[k][1]))
    lut = np.zeros(lab.max() + 1, np.int32)
    for new, k in enumerate(order, 1):
        lut[ids[k]] = new
    return lut[lab]


def sam_regions(gen, key):
    """Region map (int32, ids 1..N) at key resolution: SAM objects where SAM finds them, superpixels elsewhere."""
    s = SAM_SIDE / max(key.size)
    work = key.resize((round(key.width * s), round(key.height * s)), Image.BILINEAR)
    arr = np.asarray(work)
    masks = gen.generate(arr)
    H, W = arr.shape[:2]
    lab = np.zeros((H, W), np.int32)
    keep = [mm for mm in masks if mm['area'] >= MIN_MASK_FRAC * H * W]
    for k, mm in enumerate(sorted(keep, key=lambda mm: -mm['area']), 1):    # largest first: small ones on top
        lab[mm['segmentation']] = k
    covered = float((lab > 0).mean())
    # fill uncovered pixels with superpixels computed at 256 px and scaled up
    s2 = SLIC_SIDE / max(W, H)
    sl = slic_labels(np.asarray(work.resize((round(W * s2), round(H * s2)), Image.BILINEAR)), k=SLIC_K)
    sl = np.asarray(Image.fromarray(sl.astype(np.int32)).resize((W, H), Image.NEAREST))
    fill = lab == 0
    lab[fill] = sl[fill] + lab.max()
    lab = connected_components(lab)
    lab = merge_small(lab, min_area=MIN_REGION_FRAC * H * W, max_regions=MAX_REGIONS)
    big = Image.fromarray(lab.astype(np.int32)).resize(key.size, Image.NEAREST)
    stats = dict(sam_masks=len(masks), kept=len(keep), sam_cover=round(covered, 3))
    return np.asarray(big).astype(np.int32), stats


# --------------------------------------------------------------------------------------------- step 2
def font(size):
    for name in ('arialbd.ttf', 'arial.ttf', 'DejaVuSans-Bold.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def region_overlay(key, lab, f):
    """White 1-px region borders on a black halo, each region's number at its most interior pixel."""
    from scipy import ndimage
    a = np.asarray(key).copy()
    bd = np.zeros(lab.shape, bool)
    bd[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    bd[1:, :] |= lab[1:, :] != lab[:-1, :]
    thick = ndimage.binary_dilation(bd, iterations=1)
    a[thick] = (0, 0, 0)
    a[bd] = (255, 255, 255)
    img = Image.fromarray(a)
    d = ImageDraw.Draw(img)
    edge = bd.copy()
    edge[[0, -1], :] = True
    edge[:, [0, -1]] = True
    edt = ndimage.distance_transform_edt(~edge)
    Hh, Ww = lab.shape
    for v in range(1, int(lab.max()) + 1):
        ys, xs = np.nonzero(lab == v)
        if not len(ys):
            continue
        k = edt[ys, xs].argmax()
        t = str(v)
        x = min(max(xs[k] - 6 * len(t), 2), Ww - 13 * len(t) - 2)
        y = min(max(ys[k] - 10, 2), Hh - 24)
        d.text((x, y), t, fill='white', font=f, stroke_width=3, stroke_fill='black')
    return img


# --------------------------------------------------------------------------------------------- step 3
def load_prompt_template():
    text = PROMPT_FILE.read_text(encoding='utf8')
    if hashlib.sha256(text.encode('utf8')).hexdigest() != PROMPT_SHA256:
        print(f'WARNING: {PROMPT_FILE.name} differs from the prompt used for the training masks', file=sys.stderr)
    return text


def build_prompt(template, location, tag, n):
    """The template is a str.format string: {loc}, {tag} and {n} are filled, doubled braces become JSON braces."""
    return template.format(loc=location, tag=tag, n=n)


def claude_exe():
    """The real CLI binary (on Windows `claude` is a .cmd shim that cmd.exe would re-quote)."""
    found = shutil.which('claude')
    if found and found.lower().endswith(('.cmd', '.bat')):
        exe = Path(found).parent / 'node_modules/@anthropic-ai/claude-code/bin/claude.exe'
        if exe.exists():
            return str(exe)
    return found or 'claude'


def empty_mcp_config(out_dir):
    path = Path(out_dir) / 'empty_mcp.json'
    if not path.exists():
        path.write_text('{"mcpServers": {}}', encoding='utf8')
    return path


class UsageLimit(RuntimeError):
    pass


def ask_claude(prompt, model, query_dir, mcp_config, n, max_tries=7):
    """One headless CLI call with only the Read tool and no MCP servers; the prompt goes in on stdin and the
    working directory holds the images. Transient API errors back off and retry; two unparseable answers in
    a row give up. Returns (validated answer or None, raw result text of the last attempt)."""
    cmd = [claude_exe(), '-p', '--output-format', 'json', '--model', model,
           '--system-prompt', SYSTEM_PROMPT,
           '--strict-mcp-config', '--mcp-config', str(mcp_config),
           '--tools', 'Read', '--allowedTools', 'Read', '--no-session-persistence']
    parse_fail, result = 0, None
    for tries in range(1, max_tries + 1):
        try:
            proc = subprocess.run(cmd, input=prompt, cwd=query_dir, capture_output=True, text=True,
                                  encoding='utf8', timeout=1800)
            stdout, stderr = proc.stdout, proc.stderr
        except subprocess.TimeoutExpired:
            stdout, stderr = '', 'timeout'
        try:
            meta = json.loads(stdout)
        except json.JSONDecodeError:
            meta = {'raw_stdout': stdout[-1500:]}
        blob = (json.dumps(meta) + stderr).lower()
        if 'usage limit' in blob or 'limit reached' in blob or 'out of extra usage' in blob:
            raise UsageLimit('plan usage limit reached')
        if meta.get('is_error') or 'raw_stdout' in meta:
            if any(w in blob for w in ('rate', '429', 'overload', '529', 'timeout', 'econnreset', '500', 'api error')):
                time.sleep(min(240, 20 * 2 ** (tries - 1)) + random.random() * 10)
                continue
            print(f'  CLI call failed: {(stderr or stdout)[-300:].strip()}', file=sys.stderr)
            return None, None
        result = meta.get('result', '')
        try:
            return validate(parse_answer(result), n), result
        except Exception as exc:
            print(f'  unusable answer ({exc})', file=sys.stderr)
            parse_fail += 1
            if parse_fail >= 2:
                break
    return None, result


# --------------------------------------------------------------------------------------------- step 4
def parse_answer(text):
    """The JSON object in the model's reply (tolerates a ``` fence and text around the object)."""
    text = (text or '').strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0]
    s, e = text.find('{'), text.rfind('}')
    return json.loads(text[s:e + 1])


def validate(ans, n):
    """Reject an answer missing more than max(2, 10 %) of the n regions; record the missing ones."""
    miss = [str(i) for i in range(1, n + 1) if str(i) not in (ans.get('regions') or {})]
    if len(miss) > max(2, 0.1 * n):
        raise ValueError(f'missing regions {miss[:10]}')
    ans['_missing'] = miss
    return ans


def region_class(rec):
    """Class of one region answer: hard coral scoring <= 3 is BLEACHED, 4-6 ALIVE (an unreadable score counts
    as ALIVE), the other types by name; anything unrecognised is OTHER."""
    t = str((rec or {}).get('type', 'other')).lower() if isinstance(rec, dict) else 'other'
    if t == 'hard_coral':
        try:
            return 'BLEACHED' if float(rec.get('score', 4)) <= BLEACHED_MAX_SCORE else 'ALIVE'
        except (TypeError, ValueError):
            return 'ALIVE'
    return CLASS_OF_TYPE.get(t, 'OTHER')


# --------------------------------------------------------------------------------------------- step 5
def class_mask(lab, ans):
    """uint8 class-id mask at key resolution; regions the model did not answer stay IGNORE (255)."""
    lut = np.full(lab.max() + 1, IGNORE_ID, np.uint8)
    regs = ans.get('regions') or {}
    for v in range(1, lab.max() + 1):
        rec = regs.get(str(v))
        if rec is not None:
            lut[v] = CLASS_IDS[region_class(rec)]
    return lut[lab]


def save_mask(mask_key, size, path):
    """Nearest-upsample the key-resolution mask to the original photo size and write a mode-L PNG."""
    full = Image.fromarray(np.ascontiguousarray(mask_key, dtype=np.uint8)).resize(size, Image.NEAREST)
    full.save(path, optimize=True)


# ------------------------------------------------------------------------------------------------ driver
def tag_of(path):
    """File-name-safe tag; it names the query images and appears in the prompt."""
    return re.sub(r'[^A-Za-z0-9_-]+', '_', Path(path).stem)


def label_photo(path, gen, template, args, dirs, f):
    tag = tag_of(path)
    mask_path = dirs['masks'] / f'{tag}.png'
    if mask_path.exists() and not args.overwrite and not args.dry_run:
        print(f'{tag}: mask exists, skipped')
        return True
    t0 = time.time()
    raw, small, key = load_photo(path)
    lab, stats = sam_regions(gen, key)                                            # step 1
    n = int(lab.max())
    assert n <= 255
    Image.fromarray(lab.astype(np.uint8)).save(dirs['regions'] / f'{tag}_regions.png')
    small.save(dirs['query'] / f'{tag}_clean.jpg', quality=92)                    # what the model reads
    region_overlay(key, lab, f).save(dirs['query'] / f'{tag}_sam.jpg', quality=92, format='JPEG')   # step 2
    prompt = build_prompt(template, args.location, tag, n)
    (dirs['prompts'] / f'{tag}_prompt.txt').write_text(prompt, encoding='utf8')
    print(f'{tag}: {n} regions (SAM {stats["kept"]}/{stats["sam_masks"]} masks, cover {stats["sam_cover"]}), '
          f'{time.time() - t0:.0f}s')
    if args.dry_run:
        return True
    ans, result = ask_claude(prompt, args.model, dirs['query'], dirs['mcp'], n)  # steps 3-4
    record = dict(photo=Path(path).name, tag=tag, model=args.model, location=args.location, n_regions=n,
                  orig_size=list(raw.size), key_size=list(key.size), answer=ans, raw_result=result)
    (dirs['answers'] / f'{tag}.json').write_text(json.dumps(record, indent=1), encoding='utf8')
    if ans is None:
        print(f'{tag}: FAILED (no valid answer)')
        return False
    mask_key = class_mask(lab, ans)                                               # step 5
    save_mask(mask_key, raw.size, mask_path)
    counts = Counter(np.asarray(mask_key).ravel().tolist())
    share = {name: round(counts.get(i, 0) / mask_key.size, 3) for name, i in CLASS_IDS.items() if counts.get(i)}
    print(f'{tag}: ok, {len(ans["_missing"])} regions unanswered, area {share}, {time.time() - t0:.0f}s')
    return True


def list_photos(src):
    src = Path(src)
    if src.is_dir():
        return sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    return [src]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input', help='a photo, or a folder of photos')
    ap.add_argument('--location', required=True,
                    help='where the photos were taken; completes "...survey photo from <location>." in the prompt, '
                         'e.g. "the Red Sea" or "Norfolk Island (South Pacific)"')
    ap.add_argument('--sam-weights', required=True, help='path to sam_vit_b_01ec64.pth')
    ap.add_argument('--out', required=True, help='output folder')
    ap.add_argument('--model', default=DEFAULT_MODEL, help=f'model id passed to the claude CLI (default {DEFAULT_MODEL})')
    ap.add_argument('--dry-run', action='store_true',
                    help='build regions, overlay and the exact prompt file, but do not call the model')
    ap.add_argument('--overwrite', action='store_true', help='relabel photos that already have a mask')
    args = ap.parse_args()

    photos = list_photos(args.input)
    if not photos:
        sys.exit(f'no photos found in {args.input}')
    out = Path(args.out).resolve()        # absolute: the CLI runs with the query folder as its cwd
    dirs = {k: out / k for k in ('query', 'regions', 'prompts', 'answers', 'masks')}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    dirs['mcp'] = empty_mcp_config(out)
    template = load_prompt_template()
    gen = build_sam(args.sam_weights)
    f = font(20)
    ok = 0
    for i, p in enumerate(photos, 1):
        print(f'[{i}/{len(photos)}] {p.name}', flush=True)
        try:
            ok += bool(label_photo(p, gen, template, args, dirs, f))
        except UsageLimit as exc:
            print(f'stopping: {exc}; rerun later to continue (finished photos are skipped)')
            break
    print(f'done: {ok}/{len(photos)} photos {"prepared" if args.dry_run else "labelled"}, outputs in {out}')


if __name__ == '__main__':
    main()
