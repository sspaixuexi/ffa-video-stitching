# FFA video stitching

Algorithm code for constructing a wide-field overview from cropped intraoperative
fundus fluorescein angiography (FFA) frames. The workflow combines track-consistent
keyframe selection, local/global registration, maximum fusion, ghost detection and
source-guided correction (SGC).

The release contains algorithm source, configuration examples and tests. Clinical
images, patient identifiers, annotations, experiment outputs and trained weights
are not distributed.

## Installation

Python 3.10 is the tested environment. From the repository root:

```bash
python -m venv .venv
# Activate .venv using your shell.
python -m pip install -e ".[test]"
python -m pytest -q
```

The CPU example needs no neural-network weights:

```bash
python examples/make_toy_frames.py --output data/toy_frames
ffa-stitch --frames data/toy_frames --output outputs/toy --matcher sift --selection fixed --fixed-step 1
```

This example checks installation with artificial line patterns. SIFT is a
supported baseline; the default manuscript pipeline uses EfficientLoFTR.

## Stitch a sequence

Install a suitable PyTorch/CUDA build, the optional matching dependencies, and the
external [EfficientLoFTR](https://github.com/zju3dv/EfficientLoFTR) source. Obtain its
outdoor checkpoint from the upstream project. Keep both outside this repository
or under the ignored `third_party/` and `weights/` directories.

```bash
python -m pip install -e ".[eloftr]"
ffa-stitch --frames data/frames --output outputs/stitch --eloftr-repo third_party/EfficientLoFTR --weights weights/eloftr_outdoor.ckpt --device cuda
```

Input is a single naturally ordered directory of image files. Frames should be
cropped consistently to the relevant fundus field. The loader converts them to
grayscale and records their conversion to working-image coordinates. If starting
with a video, a separate extraction utility provides explicit frame sampling and
an optional fixed crop:

```bash
ffa-extract --video data/input.mp4 --output data/frames --stride 1 --crop 0 0 512 512
```

The crop above is only a syntax example; choose coordinates for your own video.
Extraction does not perform automatic quality selection. The default stitching
configuration performs adaptive keyframe selection afterwards.

| Setting | Default |
| --- | --- |
| Continuing-track count threshold | 15 |
| Continuing-track ratio | 0.2 |
| Componentwise tracking tolerance | 10 working pixels |
| Rematching tolerance | 20 working pixels |
| Geometric model | Similarity transform, RANSAC |
| Reprojection threshold | 5 working pixels |
| Composition | Local then global; maximum fusion |
| Canvas | Fixed |

The implementation preserves the tracking denominator, mutual-nearest-neighbor
rule, target-endpoint refinement and partition behavior of the experimental
pipeline. The last frame is not forcibly retained by default. Use `ffa-stitch
--help` for the supported controlled variants. Match caching is disabled unless
`--cache-dir` is supplied.

Outputs include panorama pieces, masks, retained working frames and
`stitch_manifest.json`. Disconnected image groups may yield multiple pieces.
The manifest records all transforms and each intermediate projection step, so
source-guided correction can reproduce the original interpolation and clipping
order. Timing records identify their included stages.

## Ghost synthesis, detection and correction

The package includes the original local-transform/max-fusion synthesis modules,
plus command-line adapters. `pretrain` selects the v3 generator used for the
synthetic pretraining set; `adaptation` selects the later v4 generator. Both use
only image files that you supply. Generated boxes describe synthetic artifacts.

```bash
ffa-ghost-synthesize --help
ffa-ghost-detect --help
ffa-ghost-correct --help
```

Detection uses an external checkout of
[official YOLOv10](https://github.com/THU-MIG/yolov10), which exposes
`ultralytics.YOLOv10`, and a user-supplied checkpoint. Install that project's
requirements in a compatible environment; a generic recent `ultralytics`
installation is not a substitute for the original YOLOv10 implementation.
Inference requires a single-class ghost-finetuned checkpoint; use generic COCO
weights only to initialize training.

SGC reads the stitch manifest, original working frames and candidate boxes in
panorama pixel coordinates. It preserves the two-stage fusion baseline and the
complete source projection route. Exterior-context registration, bounded
photometric adjustment and coverage checks determine whether a donor replaces a
candidate region. Unsupported/rejected regions retain the input; pixels outside
the edit mask remain unchanged. BSC and FCC are available as comparison policies.

See [the workflow commands](docs/ghost_workflow.md) for exact input schemas and
an end-to-end example.

## Detector training

`configs/detector_pretrain.json` and `configs/detector_finetune.json` contain the
training argument profiles used by the workflow. Supply your own labeled dataset
and initialization checkpoint. Dataset structure is illustrated in
`configs/dataset.example.yaml`; split your data before training, keeping all
images from one patient together.

```bash
ffa-ghost-train --yolov10-repo third_party/yolov10 --weights weights/yolov10m.pt --data data/pretrain.yaml --config configs/detector_pretrain.json --output outputs/pretrain
ffa-ghost-train --yolov10-repo third_party/yolov10 --weights outputs/pretrain/train/weights/best.pt --data data/real.yaml --config configs/detector_finetune.json --output outputs/finetune
```

The adapter invokes the upstream trainer with explicit paths and writes a new
run. Packaging this repository does not retrain a model or change the experiments
reported in the manuscript. Use trusted checkpoint files, since upstream PyTorch
checkpoint loading can deserialize Python objects.

## Implementation map

| Module | Purpose |
| --- | --- |
| `core.py` | Tracking, selection, refinement, partitioning and composition |
| `matching.py` | EfficientLoFTR, SIFT and ORB adapters; optional matching cache |
| `stitch_cli.py` | Portable input/output and transform manifest |
| `extract.py` | Optional video frame extraction with an explicit crop |
| `ghost/` | Synthetic ghost generation, detection and source correction |
| `detector_train.py` | Thin adapter to the external YOLOv10 trainer |

`requirements-tested.txt` records the stitching/correction dependency versions.
Third-party model implementations and licenses are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
