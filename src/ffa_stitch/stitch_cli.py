"""Run the stitching pipeline on a naturally ordered directory of frames."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .core import Config, load_sequence, stitch
from .matching import MatchEngine


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k):json_value(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [json_value(v) for v in value]
    return value


def write_png(path, image):
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise RuntimeError('PNG encoding failed: '+str(path))
    encoded.tofile(path)


def export_result(output, result, frames, paths, engine, decode_prepare_s):
    """Save images and coordinate provenance without embedding absolute paths.

    Matrices map homogeneous column vectors (x, y, 1). The geometric matrices
    in the algorithm are defined in working-image pixels. Multiplication by
    original_to_work maps the original input pixels to the same output canvas.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {k:v for k,v in result.items() if k!='pieces'}
    manifest.update(
        schema_version=1,
        coordinate_convention='homogeneous column vectors: destination = H @ source',
        matcher=engine.metadata,
        model_load_s=engine.load_s,
        decode_prepare_s=decode_prepare_s,
        frames=[dict(index=f['index'], file=p.name,
                     original_shape=f['original_shape'], working_shape=f['image'].shape,
                     original_to_work=f['original_to_work']) for f,p in zip(frames,paths)],
        pieces=[])
    started = time.perf_counter()
    # Saved working frames allow correction to repeat the original local and
    # global resampling stages, rather than interpolating a single composed warp.
    working_dir = output/'working_frames'
    working_dir.mkdir(exist_ok=True)
    for index in result['keyframes']:
        image_name = f'working_frames/frame_{index:06d}.png'
        mask_name = f'working_frames/frame_{index:06d}_mask.png'
        write_png(output/image_name,frames[index]['image'])
        write_png(output/mask_name,frames[index]['mask'])
        manifest['frames'][index].update(working_image=image_name,working_mask=mask_name)
    for j,piece in enumerate(result['pieces']):
        image_name = f'piece_{j:03d}.png'
        mask_name = f'piece_{j:03d}_mask.png'
        write_png(output/image_name, piece['image'])
        write_png(output/mask_name, piece['mask'])
        transforms = {
            str(k):dict(H_work_to_canvas=h,
                        H_original_to_canvas=h@frames[k]['original_to_work'],
                        **piece['source_geometry'][k])
            for k,h in piece['transforms'].items()
        }
        manifest['pieces'].append(dict(index=j, image=image_name, mask=mask_name,
                                       keyframes=piece['keyframes'], transforms=transforms))
    manifest['save_png_s'] = time.perf_counter()-started
    manifest['processing_s'] = decode_prepare_s+result['pipeline_s']+manifest['save_png_s']
    manifest['processing_time_definition'] = (
        'frame decoding, preparation, selection, registration, composition and PNG writing; '
        'excludes model loading and JSON writing; disk caching is optional')
    (output/'stitch_manifest.json').write_text(
        json.dumps(json_value(manifest),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames',type=Path,required=True,help='Directory of cropped grayscale FFA frames')
    parser.add_argument('--output',type=Path,required=True,help='Directory for panorama pieces and transform manifest')
    parser.add_argument('--matcher',choices=('eloftr','sift','orb'),default='eloftr')
    parser.add_argument('--eloftr-repo',type=Path,help='External EfficientLoFTR source checkout')
    parser.add_argument('--weights',type=Path,help='External EfficientLoFTR checkpoint')
    parser.add_argument('--device',default='cuda',help='EfficientLoFTR device, e.g. cuda, cuda:0 or cpu')
    parser.add_argument('--cache-dir',type=Path,help='Optional correspondence cache; disabled when omitted')
    parser.add_argument('--coarse-threshold',type=float,default=0.7)
    parser.add_argument('--selection',choices=('adaptive','fixed','motion'),default='adaptive')
    parser.add_argument('--fixed-step',type=int,default=2)
    parser.add_argument('--min-tracks',type=int,default=15)
    parser.add_argument('--track-ratio',type=float,default=0.2)
    parser.add_argument('--track-tolerance',type=float,default=10)
    parser.add_argument('--rematch-tolerance',type=float,default=20)
    parser.add_argument('--no-refine',action='store_true')
    parser.add_argument('--no-partition',action='store_true')
    parser.add_argument('--include-last',action='store_true',help='Explicitly include final input frame; disabled by default')
    parser.add_argument('--transform',choices=('similarity','affine','homography'),default='similarity')
    parser.add_argument('--canvas',choices=('fixed','dynamic'),default='fixed')
    parser.add_argument('--fusion',choices=('max','masked_max','feather','multiband','graphcut'),default='max')
    parser.add_argument('--max-canvas-pixels',type=int,default=20_000_000)
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--seed',type=int,default=20260916)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.fixed_step<1 or args.min_tracks<0 or args.threads<1:
        parser.error('fixed-step and threads must be positive; min-tracks must be nonnegative')
    if not 0<=args.track_ratio<=1 or not 0<=args.coarse_threshold<=1:
        parser.error('track-ratio and coarse-threshold must be in [0, 1]')
    if args.track_tolerance<0 or args.rematch_tolerance<0 or args.max_canvas_pixels<1:
        parser.error('tolerances must be nonnegative and max-canvas-pixels positive')
    if args.matcher=='eloftr' and (args.eloftr_repo is None or args.weights is None):
        parser.error('eloftr requires --eloftr-repo and --weights')
    if not args.frames.is_dir():
        parser.error('frames must be an existing directory')
    if args.output.resolve()==args.frames.resolve():
        parser.error('output must differ from the input frames directory')
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error('output must be a new or empty directory')
    cv2.setNumThreads(args.threads)
    if args.matcher=='eloftr':
        import torch
        torch.set_num_threads(args.threads)
    cfg = Config(selection=args.selection, fixed_step=args.fixed_step,
                 min_tracks=args.min_tracks, track_ratio=args.track_ratio,
                 track_tolerance=args.track_tolerance, rematch_tolerance=args.rematch_tolerance,
                 refine_matches=not args.no_refine, partition=not args.no_partition,
                 include_last=args.include_last, transform=args.transform, canvas=args.canvas,
                 fusion=args.fusion, max_canvas_pixels=args.max_canvas_pixels, seed=args.seed)
    engine = MatchEngine(args.matcher,coarse=args.coarse_threshold,
                         cache_dir=args.cache_dir,eloftr_repo=args.eloftr_repo,
                         weights=args.weights,device=args.device)
    started = time.perf_counter()
    frames,paths = load_sequence(args.frames)
    decode_prepare_s = time.perf_counter()-started
    result = stitch(frames,engine,cfg)
    manifest = export_result(args.output,result,frames,paths,engine,decode_prepare_s)
    print(json.dumps(dict(inputs=len(frames),keyframes=len(result['keyframes']),
                          pieces=len(result['pieces']),processing_s=manifest['processing_s']),indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
