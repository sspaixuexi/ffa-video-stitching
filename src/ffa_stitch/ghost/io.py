"""Portable, non-overwriting I/O shared by the optional ghost commands."""
from pathlib import Path
import hashlib
import json

import cv2
import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    def convert(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        raise TypeError(type(item).__name__)
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False, default=convert)
        handle.write('\n')


def read_image(path, grayscale=False):
    """Read 8-bit gray or color input, returning gray or OpenCV BGR."""
    image = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f'Cannot decode image: {Path(path).name}')
    if image.dtype != np.uint8 or image.ndim not in (2, 3):
        raise ValueError('Expected an 8-bit gray or three-channel image')
    if image.ndim == 3 and image.shape[2] != 3:
        raise ValueError('Expected three channels without an alpha channel')
    if grayscale:
        return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image


def write_png(path, image, rgb=False):
    if image.dtype == bool:
        image = image.astype(np.uint8) * 255
    if rgb and image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise OSError('PNG encoding failed')
    with Path(path).open('xb') as handle:
        handle.write(encoded.tobytes())


def new_directory(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError('Output directory must be empty or new')
    path.mkdir(parents=True, exist_ok=True)
    return path


def manifest_file(root, name):
    """Stitch artifacts are relative to, and contained in, their manifest directory."""
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError('Stitch artifact paths must be nonempty relative paths')
    path = (Path(root) / name).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('Stitch artifact path leaves its manifest directory')
    if not path.is_file():
        raise ValueError(f'Missing stitch artifact: {name}')
    return path
