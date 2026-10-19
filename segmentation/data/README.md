# Training data

The trainer reads two kinds of data. Neither is stored in this repository.

## 1. CoralScapes V2 (public)

Red Sea reef images with dense labels, from Hugging Face (`josauder/coralscapesV2`), ~7 GB download.

```bash
pip install huggingface_hub pyarrow pyyaml
python data/coralscapes/download_v2.py        # -> data/coralscapes_v2/hf/
python data/coralscapes/decode_v2.py          # -> data/coralscapes_v2/processed/{train,validation,test}_512/
```
Override the location with `CORAL_DATA_V2=/path/to/coralscapes_v2`. `health_label_mapping_v2.yaml` maps the
95 CoralScapes classes to 4 health classes (LIVE, BLEACHED, DEGRADED, OTHER). The 7-class trainer uses those
as partial labels: each 4-class label allows a set of 7-class answers.

Optional but much faster: build the decoded cache once. The trainer builds it on first use when `--cache DIR`
or `CORAL_V2_CACHE` is set; without it every crop decodes a PNG and the profile check fails.

## 2. Our 7-class labels (on request)

2,244 CoralNet survey photos from six regions, labelled with the pipeline in `../labeller/`. The masks were
made locally and are not published here. **To retrain, ask the team for the label exports.** Put them in
`data/labels/` (or point `LABELS=` at them):

```
data/labels/
  sonnet7_kenya/            870 training photos
  sonnet7_red_sea/          577
  sonnet7_chagos/           573
  sonnet7_la_paz/           135
  sonnet7_norfolk/           41
  sonnet7_raja/              48   (Raja Ampat: training only)
  sonnet7_<region>_heldout/      validation: photos from held-out reef sites, one folder per region except Raja
```
Folder layout and pixel format: [EXPORT_FORMAT.md](EXPORT_FORMAT.md). None of these photos are in the
benchmark: the benchmark photos come from other held-out sites and were excluded from every export.
