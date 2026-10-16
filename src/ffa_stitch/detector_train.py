"""Train an external official YOLOv10 model with an explicit local dataset."""
import argparse
import importlib
import json
from pathlib import Path
import sys


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--yolov10-repo', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True,
                   help='Trusted upstream initialization or a previous training checkpoint')
    p.add_argument('--data', type=Path, required=True, help='User-supplied YOLO dataset YAML')
    p.add_argument('--config', type=Path, required=True, help='Training arguments as JSON')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='0')
    a = p.parse_args(argv)
    for name in ('weights', 'data', 'config'):
        if not getattr(a, name).is_file():
            p.error(f'{name} file does not exist')
    if not (a.yolov10_repo/'ultralytics').is_dir():
        p.error('Expected the official YOLOv10 source checkout')
    if a.output.exists() and any(a.output.iterdir()):
        p.error('Use an empty output directory to preserve previous training runs')
    options = json.loads(a.config.read_text(encoding='utf-8'))
    if not isinstance(options, dict):
        p.error('Training config must be a JSON object')
    reserved = {'data', 'model', 'project', 'name', 'device', 'resume', 'exist_ok'}
    if reserved & options.keys():
        p.error('Config contains reserved arguments: '+', '.join(sorted(reserved & options.keys())))
    sys.path.insert(0, str(a.yolov10_repo.resolve()))
    upstream = importlib.import_module('ultralytics')
    if not Path(upstream.__file__).resolve().is_relative_to(a.yolov10_repo.resolve()):
        raise RuntimeError('An unrelated ultralytics installation was imported; use a fresh process/environment')
    if not hasattr(upstream, 'YOLOv10'):
        raise RuntimeError('This checkout does not expose official YOLOv10')
    # Training occurs only when this command is explicitly called with local data.
    model = upstream.YOLOv10(str(a.weights))
    model.train(data=str(a.data.resolve()), project=str(a.output.resolve()), name='train',
                device=a.device, resume=False, exist_ok=False, **options)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
