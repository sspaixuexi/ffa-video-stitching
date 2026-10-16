# Ghost workflow

Run the commands from the repository root after installing the package. Output
locations must be new or empty so that previous runs remain intact.

## Panorama detection and correction

First produce `outputs/stitch/stitch_manifest.json` with `ffa-stitch`. Install
the official YOLOv10 dependencies in the detector environment. Then run:

```bash
ffa-ghost-detect --image outputs/stitch/piece_000.png --checkpoint weights/ghost_detector.pt --yolov10-repo third_party/yolov10 --output outputs/boxes.json --device 0
ffa-ghost-correct --manifest outputs/stitch/stitch_manifest.json --piece-index 0 --boxes outputs/boxes.json --output outputs/corrected --strategy SGC
```

Detection uses the one-to-one YOLOv10 head, a default confidence threshold of
0.25, a 1,024-pixel inference size and no additional NMS. A local checkpoint is
required. It must be a single-class checkpoint fine-tuned for `ghost` (or
`ghosting`); original COCO weights are a training initialization, not a ghost
detector. If a sequence produces several panorama pieces, detect and correct
each piece separately, keeping its image, piece index and boxes together.

The detector writes native panorama coordinates in a JSON document:

```json
{
  "schema_version": "ffa-ghost-boxes-1",
  "coordinate_space": "piece_canvas_pixels",
  "image_size_wh": [640, 480],
  "boxes": [
    {"xyxy": [100, 120, 180, 190], "score": 0.82, "class_id": 0}
  ]
}
```

This is a schema example with arbitrary dimensions and coordinates, not an
annotation from the study. Actual detector output also binds the image and
checkpoint by SHA-256. The correction loader rejects mismatched image dimensions
or hashes. `boxes: []` is a no-box identity case.

`--strategy SGC` uses exterior-context source selection and bounded local
adjustment; `BSC` selects by brightness; `FCC` requires full region coverage.
Outputs are `corrected.png`, `edit_mask.png`, `core_mask.png` and
`correction.json`. The JSON records accepted/rejected candidate regions, donor
selection and the core processing time. It does not treat a technically accepted
region as an independently measured clinical outcome.

The stitch output includes retained grayscale working frames and their masks.
Each source has a `projection_route` containing every ordered transform and its
destination canvas size. SGC projects donors through this route with linear
image interpolation and nearest-neighbor mask interpolation. It reads the saved
panorama as the baseline, preserving maximum fusion followed by subsequent
global interpolation. Do not replace this with one warp of the original frame
or a maximum of already projected donors.

## Generate synthetic training images

Create a source manifest, for example `data/sources.json`:

```json
{
  "sources": [
    {
      "id": "source_a",
      "image": "images/source_a.png",
      "group": "group_a",
      "split": "train",
      "dataset": "user_dataset",
      "domain": "public"
    },
    {
      "id": "source_b",
      "image": "images/source_b.png",
      "group": "group_b",
      "split": "val",
      "dataset": "user_dataset",
      "domain": "public"
    }
  ]
}
```

Image paths are relative to the manifest. Use an anonymous `group` to keep all
images from one patient or source together. A group or identical source image
cannot cross the train/validation split. An optional `mask` names a vessel-mask
image; otherwise the generator estimates bright line structures. Optional
`crop_xyxy` provides an explicit source crop.

```bash
ffa-ghost-synthesize --sources data/sources.json --output outputs/synthetic --profile pretrain --count 5000
```

The `pretrain` profile retains the v3 geometry, visibility and intensity rules.
The `adaptation` profile retains v4 variable-region synthesis. Local-domain
adaptation sources must use `domain: "local"` and `split: "train"`.

Outputs contain `images/`, `labels/`, per-sample metadata, rejection records and
small QA previews. A YOLO `data.yaml` is produced only if accepted samples exist
in both splits. The requested count is a target; visibility and content gates
may reject proposals. The command exits with a nonzero status and reports quota
deficits if it cannot reach the target within the proposal limit. Review the
generated examples before training. Synthetic image counts and split ratios
depend on the supplied source manifest; the wrapper does not reconstruct the
private study dataset or promise its exact sample allocation.
