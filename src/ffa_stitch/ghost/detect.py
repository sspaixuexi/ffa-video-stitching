"""YOLOv10 one-to-one detection on a user-supplied panorama and checkpoint.

Install the official YOLOv10 project separately. Its implementation and weights
are not bundled here. Output boxes use native panorama pixels and feed correct.
"""
import argparse
import importlib
from pathlib import Path
import sys

import numpy as np

from .io import read_image, sha256, write_json


def load_yolov10_class(repository=None):
    """Return external ultralytics.YOLOv10, optionally from an explicit checkout."""
    repo = Path(repository).resolve() if repository is not None else None
    if repo is not None:
        if not (repo/'ultralytics/__init__.py').is_file():
            raise ValueError('yolov10-repo must contain ultralytics/__init__.py')
        sys.path.insert(0, str(repo))
    try:
        module = importlib.import_module('ultralytics')
    except ImportError as error:
        raise RuntimeError('Install the official YOLOv10 repository and its dependencies separately') from error
    if repo is not None and not Path(module.__file__).resolve().is_relative_to(repo):
        raise RuntimeError('A different ultralytics was already imported; start a fresh process')
    model_class = getattr(module, 'YOLOv10', None)
    if model_class is None:
        raise RuntimeError('This ultralytics does not expose YOLOv10; use the official YOLOv10 checkout')
    return model_class


def select_boxes(xyxy, scores, classes, width, height, confidence=.25):
    """Filter confidence >= threshold with no additional NMS or box merging."""
    xyxy = np.asarray(xyxy, dtype=float).reshape(-1, 4)
    scores, classes = np.asarray(scores, float).reshape(-1), np.asarray(classes, float).reshape(-1)
    if len(xyxy) != len(scores) or len(scores) != len(classes):
        raise ValueError('Detector box/score/class lengths differ')
    if not np.isfinite(xyxy).all() or not np.isfinite(scores).all() or not np.isfinite(classes).all():
        raise ValueError('Detector output contains nonfinite values')
    selected = []
    for b, score, cls in zip(xyxy, scores, classes):
        if cls != 0 or score < confidence:
            continue
        b = np.clip(b, [0, 0, 0, 0], [width, height, width, height])
        if b[2] <= b[0] or b[3] <= b[1]:
            continue
        selected.append({'xyxy': b.tolist(), 'score': float(score), 'class_id': 0})
    return selected


def run(image, checkpoint, output, yolov10_repo=None, confidence=.25, image_size=1024, device='cpu'):
    image, checkpoint, output = Path(image), Path(checkpoint), Path(output)
    if not checkpoint.is_file():
        raise ValueError('Supply a local checkpoint file; automatic weight downloads are disabled')
    if output.exists():
        raise ValueError('Output JSON already exists')
    if not 0 < confidence <= 1 or image_size < 32:
        raise ValueError('confidence must be in (0,1] and image-size at least 32')
    bgr = read_image(image)
    height, width = bgr.shape[:2]
    model = load_yolov10_class(yolov10_repo)(str(checkpoint))
    names = model.names
    values = list(names.values()) if isinstance(names, dict) else list(names)
    if len(values) != 1 or str(values[0]).strip().lower() not in ('ghost', 'ghosting'):
        raise ValueError('Detection requires a single-class ghost-finetuned checkpoint; COCO weights are only an initialization for training')
    # Upstream predict uses a strict > conf filter. Use a zero floor below the
    # requested positive threshold, then explicitly apply >= on one-to-one
    # predictions. There is no extra NMS and at most 300 detections per image.
    results = model.predict(source=bgr, imgsz=image_size, conf=0.0, max_det=300,
                            device=device, half=False, save=False, verbose=False)
    if len(results) != 1:
        raise ValueError('Expected one prediction result for one panorama')
    result = results[0]
    if result.boxes is None:
        selected = []
    else:
        box = result.boxes
        selected = select_boxes(box.xyxy.detach().cpu().numpy(), box.conf.detach().cpu().numpy(),
                                box.cls.detach().cpu().numpy(), width, height, confidence)
    report = {'schema_version': 'ffa-ghost-boxes-1', 'coordinate_space': 'piece_canvas_pixels',
              'image_size_wh': [width, height], 'image_sha256': sha256(image),
              'checkpoint_sha256': sha256(checkpoint), 'confidence_threshold': confidence,
              'image_size': image_size, 'max_detections': 300, 'extra_nms': False,
              'class_names': {'0': 'ghost'}, 'boxes': selected}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--yolov10-repo', type=Path)
    parser.add_argument('--confidence', type=float, default=.25)
    parser.add_argument('--image-size', type=int, default=1024)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    try:
        report = run(args.image, args.checkpoint, args.output, args.yolov10_repo,
                     args.confidence, args.image_size, args.device)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(2, f'Detection failed: {error}\n')
    print(f"{len(report['boxes'])} candidate boxes written to {args.output}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
