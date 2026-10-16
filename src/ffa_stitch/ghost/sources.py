"""Project single working frames through saved, ordered stitching operations.

The existing panorama is the baseline. It is never reconstructed as the maximum
of these individual donors: maximum fusion and subsequent interpolation do not
commute. Coordinate matrices alone cannot preserve intermediate canvas clipping.
"""
from pathlib import Path

import cv2
import numpy as np

from .io import manifest_file, read_image, read_json, sha256


def matrix(value):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3, 3) or not np.isfinite(result).all():
        raise ValueError('Expected a finite 3x3 transform')
    if abs(np.linalg.det(result)) < 1e-12:
        raise ValueError('Singular source transform')
    return result


def warp(image, transform, size, nearest=False):
    flags = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    if np.allclose(transform[2], [0, 0, 1], atol=1e-12):
        return cv2.warpAffine(image, transform[:2], size, flags=flags)
    return cv2.warpPerspective(image, transform, size, flags=flags)


def content_fov(work):
    """Low-intensity filled FOV heuristic; this is not a vessel segmentation."""
    if work.dtype != np.uint8 or work.ndim != 2:
        raise ValueError('Content mask requires an 8-bit grayscale working image')
    smooth = cv2.GaussianBlur(work, (0, 0), 1.5, borderType=cv2.BORDER_REFLECT_101)
    binary = (smooth > 5).astype(np.uint8) * 255
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8),
                              iterations=1, borderType=cv2.BORDER_REPLICATE)
    contours = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    keep = [c for c in contours if cv2.contourArea(c) >= work.size * .01]
    mask = np.zeros(work.shape, np.uint8)
    if keep:
        cv2.drawContours(mask, keep, -1, 255, cv2.FILLED)
    return mask


def project(work, coarse_mask, route):
    """Keep each warp (including identity), interpolation and intermediate size."""
    if not isinstance(route, list) or not route:
        raise ValueError('Each donor needs a nonempty projection_route')
    image = work.copy()
    content = content_fov(work)
    coarse = coarse_mask.copy()
    support = np.full(work.shape, 255, np.uint8)
    cumulative = np.eye(3)
    for operation in route:
        h = matrix(operation['matrix'])
        size = operation['canvas_size_wh']
        if len(size) != 2 or any(isinstance(v, bool) or int(v) != v or v < 1 for v in size):
            raise ValueError('Projection canvas_size_wh must be positive integer [width,height]')
        size = tuple(map(int, size))
        image = warp(image, h, size)
        content = warp(content, h, size, nearest=True)
        coarse = warp(coarse, h, size, nearest=True)
        support = warp(support, h, size, nearest=True)
        cumulative = h @ cumulative
    content = (content > 0) & (support > 0)
    coarse = (coarse > 0) & (support > 0)
    return np.repeat(image[..., None], 3, axis=2), content, coarse, support > 0, cumulative


def load_piece(manifest_path, piece_index=0):
    manifest_path = Path(manifest_path)
    manifest = read_json(manifest_path)
    pieces = manifest['pieces']
    if not 0 <= piece_index < len(pieces):
        raise ValueError('piece-index is outside the manifest pieces list')
    piece = pieces[piece_index]
    root = manifest_path.parent
    baseline_path = manifest_file(root, piece['image'])
    baseline = cv2.cvtColor(read_image(baseline_path), cv2.COLOR_BGR2RGB)
    frames = {int(f['index']): f for f in manifest['frames']}
    if len(frames) != len(manifest['frames']):
        raise ValueError('Duplicate frame indices')
    transforms = piece['transforms']
    selected = piece.get('keyframes', [int(k) for k in transforms])
    if len(set(selected)) != len(selected) or {str(k) for k in selected} != set(transforms):
        raise ValueError('Piece keyframes and source transforms disagree')
    donors, content, coarse, support, provenance = [], [], [], [], []
    for index in selected:
        frame, geometry = frames[int(index)], transforms[str(index)]
        work_path = manifest_file(root, frame['working_image'])
        coarse_path = manifest_file(root, frame['working_mask'])
        work = read_image(work_path, grayscale=True)
        coarse_work = read_image(coarse_path, grayscale=True)
        if coarse_work.shape != work.shape:
            raise ValueError('Working image/mask dimensions disagree')
        if frame.get('working_shape') is not None and tuple(frame['working_shape'][:2]) != work.shape:
            raise ValueError('Working image dimensions disagree with manifest')
        if 'projection_route' not in geometry:
            raise ValueError('Missing ordered projection_route; a composed transform is insufficient')
        donor, cmask, fmask, smask, h = project(work, coarse_work, geometry['projection_route'])
        if donor.shape != baseline.shape:
            raise ValueError('Projected donor canvas differs from the saved panorama')
        if not np.allclose(h, matrix(geometry['H_work_to_canvas']), rtol=1e-10, atol=1e-9):
            raise ValueError('Projection route disagrees with composed working transform')
        if 'H_original_to_canvas' in geometry and not np.allclose(
                h @ matrix(frame['original_to_work']), matrix(geometry['H_original_to_canvas']),
                rtol=1e-10, atol=1e-9):
            raise ValueError('Original-to-canvas coordinate mapping disagrees')
        donors.append(donor); content.append(cmask); coarse.append(fmask); support.append(smask)
        provenance.append({'frame_index': int(index), 'working_image_sha256': sha256(work_path),
                           'working_mask_sha256': sha256(coarse_path),
                           'projection_route': geometry['projection_route']})
    shape = baseline.shape[:2]
    return {'baseline': baseline, 'baseline_sha256': sha256(baseline_path),
            'donors': np.stack(donors) if donors else np.empty((0, *shape, 3), np.uint8),
            'content_masks': np.stack(content) if content else np.empty((0, *shape), bool),
            'coarse_masks': np.stack(coarse) if coarse else np.empty((0, *shape), bool),
            'support_masks': np.stack(support) if support else np.empty((0, *shape), bool),
            'donor_ids': [str(k) for k in selected], 'provenance': provenance}
