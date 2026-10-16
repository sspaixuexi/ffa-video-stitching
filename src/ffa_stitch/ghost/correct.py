"""Apply SGC (default), BSC or FCC to one saved final stitching piece.

Example: python -m ffa_stitch.ghost.correct --manifest run/stitch_manifest.json
         --piece-index 0 --boxes boxes.json --output corrected
"""
import argparse
from pathlib import Path

import numpy as np

from . import full_coverage, postprocess
from .io import new_directory, read_json, sha256, write_json, write_png
from .sources import load_piece


def read_boxes(path, image_shape, baseline_sha256=None):
    data = read_json(path)
    if not isinstance(data, dict) or data.get('coordinate_space') != 'piece_canvas_pixels':
        raise ValueError('Boxes must explicitly use coordinate_space=piece_canvas_pixels')
    expected = [int(image_shape[1]), int(image_shape[0])]
    if data.get('image_size_wh') != expected:
        raise ValueError('Box image_size_wh differs from the selected panorama canvas')
    if data.get('image_sha256') and baseline_sha256 != data['image_sha256']:
        raise ValueError('Boxes belong to another image file (SHA-256 mismatch)')
    boxes = []
    for entry in data['boxes']:
        if isinstance(entry, dict):
            if entry.get('class_id', 0) != 0:
                continue
            entry = entry['xyxy']
        b = np.asarray(entry, float)
        if b.shape != (4,) or not np.isfinite(b).all() or b[2] <= b[0] or b[3] <= b[1]:
            raise ValueError('Each box must be finite, ordered [x1,y1,x2,y2]')
        boxes.append(b.tolist())
    return boxes


def run(manifest, boxes_file, output, piece_index=0, strategy='SGC'):
    if strategy not in ('SGC', 'BSC', 'FCC'):
        raise ValueError('strategy must be SGC, BSC or FCC')
    if Path(output).exists() and any(Path(output).iterdir()):
        raise ValueError('Output directory must be empty or new')
    inputs = load_piece(manifest, piece_index)
    baseline = inputs['baseline']
    boxes = read_boxes(boxes_file, baseline.shape, inputs['baseline_sha256'])
    if strategy == 'FCC':
        result = full_coverage.correct(baseline, inputs['donors'], inputs['coarse_masks'],
                                       boxes, inputs['donor_ids'], policy='R1')
        parameters = full_coverage.PARAMETERS
    else:
        result = postprocess.postprocess(baseline, inputs['donors'], inputs['content_masks'],
                                         inputs['support_masks'], boxes, inputs['donor_ids'],
                                         mode='C' if strategy == 'SGC' else 'B')
        parameters = postprocess.PARAMETERS
    output = new_directory(output)
    write_png(output/'corrected.png', result['image'], rgb=True)
    write_png(output/'edit_mask.png', result['edit_mask'])
    write_png(output/'core_mask.png', result['core_mask'])
    report = {'schema_version': 'ffa-ghost-correction-1', 'strategy': strategy,
              'piece_index': piece_index, 'manifest_sha256': sha256(manifest),
              'boxes_sha256': sha256(boxes_file), 'baseline_sha256': inputs['baseline_sha256'],
              'parameters': parameters, 'summary': result['summary'], 'regions': result['log'],
              'donors': inputs['provenance'], 'coordinate_space': 'piece_canvas_pixels',
              'donor_projection': 'ordered_saved_operations_linear_image_nearest_mask',
              'timing_scope': 'correction core only; excludes detector, projection and I/O',
              'outputs': {'image': 'corrected.png', 'edit_mask': 'edit_mask.png', 'core_mask': 'core_mask.png'}}
    write_json(output/'correction.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--boxes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--piece-index', type=int, default=0)
    parser.add_argument('--strategy', choices=['SGC', 'BSC', 'FCC'], default='SGC')
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error('threads must be positive')
    import cv2
    cv2.setNumThreads(args.threads)
    try:
        report = run(args.manifest, args.boxes, args.output, args.piece_index, args.strategy)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f'Correction failed: {error}\n')
    print(f"{args.strategy}: {report['summary']['accepted_regions']} accepted candidate regions; "
          f"{report['summary']['changed_pixels']} changed pixels")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
