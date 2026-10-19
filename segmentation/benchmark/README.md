# Bleaching benchmark (100 photos)

100 public CoralNet survey photos, 20 per region:
- Kenya (CoralNet source 4963)
- Red Sea (5546)
- Chagos (6447)
- Norfolk Island (8489)
- La Paz (5542, 6054)

Every photo comes from a **reef site that was held out of training, validation and model selection**.

| file | what |
|---|---|
| `benchmark.json` | per photo: region, CoralNet source and image id, page URL, size, SHA-256 of the original, mask path |
| `masks/<region>/<image_id>.png` | hand-corrected 7-class masks (ids 1–7 = BLEACHED, DEAD, ALIVE, ROCK, OCEAN, SAND, OTHER; 255 = ignore) |
| `fetch_photos.py` | downloads the original photos from CoralNet and verifies each one against its SHA-256 |

The photos belong to their CoralNet sources and are not redistributed here. `python benchmark/fetch_photos.py`
puts them in `benchmark/photos/` (~270 MB, git-ignored).

**How the masks were made.** The labeller in `../labeller/` pre-filled each photo. A team member then
corrected it region by region in our labelling editor. Masks are stored at the resolution they were edited at
(2048 px on the long side for most photos). Resize them to the photo size with nearest-neighbour before use;
`eval/evaluate.py` does this.

**Rules.** Test only: never train on these photos, tune thresholds on them, or pick checkpoints with them.
Report pooled scores and per-region scores, and the mean over training seeds.
