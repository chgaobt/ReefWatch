# Coral photo labeller (SAM regions + LLM, "M7C")

This folder holds the labelling pipeline that produced the 7-class training masks for the milestone
segmentation model. It turns a reef survey photo into a dense class mask without hand painting:

1. **Regions.** The photo is resized to a 1280 px long side. SAM v1 (ViT-B) automatic mask generation cuts
   it into object masks. Gaps SAM leaves are filled with superpixels, and small regions are merged until
   there are at most 80.
2. **Overlay.** The regions are outlined and numbered on the photo.
3. **Ask.** An LLM (Claude Sonnet, called through the `claude` command-line tool) reads the photo and the
   numbered overlay. For each region it answers what mostly covers it (`hard_coral`, `dead_coral`, `rock`,
   `sand`, `water` or `other`). For hard coral it also gives a CoralWatch Coral Health Chart brightness
   score from 1 (white) to 6 (dark). The exact prompt is in `prompt_m7c.txt`.
4. **Parse.** The JSON answer is validated: at most max(2, 10 %) of the regions may be missing, otherwise
   the answer is rejected. Each region answer is then mapped to a class.
5. **Mask.** Every region is painted with its class id. The mask is upsampled (nearest neighbour) to the
   original photo size and saved as a single-channel PNG.

## Classes

| id  | class    | region answer                          |
|-----|----------|----------------------------------------|
| 1   | BLEACHED | `hard_coral`, CoralWatch score <= 3    |
| 2   | DEAD     | `dead_coral`                           |
| 3   | ALIVE    | `hard_coral`, CoralWatch score 4-6     |
| 4   | ROCK     | `rock`                                 |
| 5   | OCEAN    | `water`                                |
| 6   | SAND     | `sand`                                 |
| 7   | OTHER    | `other` (or any unrecognised type)     |
| 255 | IGNORE   | region the model did not answer        |

Id 0 is never written. A `hard_coral` answer with a missing or unreadable score counts as ALIVE.

## Requirements

- Python 3.9+ with `numpy`, `pillow`, `scipy`, `opencv-python`, `torch` and `segment_anything`
  (`pip install git+https://github.com/facebookresearch/segment-anything.git`)
- SAM ViT-B weights: <https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth>
- The `claude` CLI installed and logged in (it is run headlessly, with only its Read tool enabled
  and no MCP servers). Not needed for `--dry-run`.

SAM runs on a CUDA GPU when one is available and falls back to CPU, which is much slower.

## Usage

```bash
# label a folder of photos
python label_photos.py photos/ --location "Norfolk Island (South Pacific)" \
    --sam-weights sam_vit_b_01ec64.pth --out labels/

# one photo: build the regions, the overlay and the exact prompt, without calling the model
python label_photos.py photo.jpg --location "the Red Sea" \
    --sam-weights sam_vit_b_01ec64.pth --out labels/ --dry-run
```

`--location` completes the prompt's first sentence: "...an underwater survey photo from `<location>`."
The training masks used these values: `the Kenyan coast (western Indian Ocean)`,
`Raja Ampat, Indonesia`, `the Red Sea`, `the Chagos Archipelago (central Indian Ocean)`,
`Norfolk Island (South Pacific)` and `La Paz, Gulf of California, Mexico`.
`--model` defaults to `claude-sonnet-5-5`, the model the training masks were made with. Photos that
already have a mask are skipped unless you pass `--overwrite`.

Outputs under `--out`, per photo (`<tag>` is the file name stem):

| path                               | contents                                                    |
|------------------------------------|-------------------------------------------------------------|
| `masks/<tag>.png`                  | class-id mask at the original photo size (mode L)           |
| `answers/<tag>.json`               | the parsed answer, the model's raw reply, sizes, model, location |
| `regions/<tag>_regions.png`        | region ids 1..N at the 1280 px key size                     |
| `prompts/<tag>_prompt.txt`         | the exact prompt sent                                       |
| `query/<tag>_clean.jpg`, `_sam.jpg`| the two images the model reads (the CLI runs in `query/`)   |

Photos are read as raw pixels and EXIF orientation is not applied (as in the original pipeline), so masks
line up with the stored pixel grid, not with a rotated view.

`prompt_m7c.txt` is a Python `str.format` template: `{loc}`, `{tag}` and `{n}` are filled in, and the
doubled braces become the JSON example's braces. The script warns if the file no longer matches the prompt
the training masks were made with.

## Reproducibility

Re-running this labeller will **not** reproduce the exact training masks. The model's answers vary from
run to run, and SAM output can differ slightly between hardware and library versions. The masks are
generated labels, not verified ground truth: they were not reviewed photo by photo, and rock and dead coral
are sometimes confused. The original training masks are available from the team on request.
