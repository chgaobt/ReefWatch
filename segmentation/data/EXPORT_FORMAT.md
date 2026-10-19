# Label export format

One folder per region and split, e.g. `sonnet7_kenya/` (training) and `sonnet7_kenya_heldout/` (validation).

```
processed/train_512/{images,masks,health}/NNNN.png     training folders: every photo is here
processed/validation_512/{images,masks,health}/        (empty)
processed/test_512/{images,masks,health}/NNNN.png      *_heldout folders: every photo is here
metadata/coralscapes_v2_index.csv                       one row per photo
metadata/export_info.json                               provenance, geometry, pixel counts
classes.json                                            class names and ids
manifest.json / manifest.csv                            idx, split, CoralNet image id, source, site
READY.json                                              written last; the folder is complete when it exists
```

## Pixels

- `images/NNNN.png`: RGB uint8, the full-resolution CoralNet original (raw pixels, no EXIF rotation), resized
  so the long side is 1024 px (bilinear), centre-padded to a multiple of 256 with the ImageNet mean pixel
  (124, 116, 104).
- `masks/NNNN.png`: single-channel uint8, same size as the image:

  | id | class | labeller source |
  |---|---|---|
  | 1 | BLEACHED | hard coral with CoralWatch score ≤ 3 |
  | 2 | DEAD | dead coral |
  | 3 | ALIVE | hard coral with CoralWatch score 4–6 |
  | 4 | ROCK | rock |
  | 5 | OCEAN | open water |
  | 6 | SAND | sand |
  | 7 | OTHER | anything else (fish, gear, algae, soft coral, ...) |
  | 255 | IGNORE | padding, and regions the labeller gave no answer for |
  | 0 | unlabeled | not used |

  The model's output index is the mask id minus 1.
- `health/NNNN.png`: the 4-class collapse (LIVE 0, BLEACHED 1, DEGRADED 2 = DEAD + ROCK, OTHER 3, IGNORE 255).
