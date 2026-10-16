"""Generate local max-fusion ghost candidates and YOLO labels from user images.

The pretrain profile retains the v3 generator used for public pretraining;
adaptation retains the v4 variable-region paired-label generator. Geometry,
visibility gates and intensity rules are unchanged. Inputs are a user manifest:
{"sources": [{"id":"image_a", "image":"images/a.png", "group":"case_a",
               "split":"train", "dataset":"custom", "domain":"public"}]}.
Optional fields: mask (vessel-mask filename), crop_xyxy (explicit safety crop).
Paths are relative to the source manifest. Group and image-hash split conflicts
are rejected. This command does not choose a patient split for the user.
"""
import argparse
from collections import Counter, OrderedDict
import importlib
import json
from pathlib import Path

import cv2
import numpy as np

from .io import new_directory, read_image, read_json, sha256, write_json, write_png


def load_sources(manifest_path, profile):
    manifest_path = Path(manifest_path)
    data = read_json(manifest_path)
    records, seen_ids, group_splits, image_splits = [], set(), {}, {}
    for index, record in enumerate(data['sources']):
        entry = dict(record)
        if not isinstance(entry.get('id'), str) or not entry['id'] or entry['id'] in seen_ids:
            raise ValueError('Every source requires a unique, nonempty string id')
        seen_ids.add(entry['id'])
        if not isinstance(entry.get('group'), str) or not entry['group']:
            raise ValueError('Every source requires an explicit group identifier')
        if entry.get('split') not in ('train', 'val'):
            raise ValueError('Every source requires split=train or split=val')
        if entry.get('domain', 'public') not in ('public', 'local'):
            raise ValueError('domain must be public or local')
        if profile == 'adaptation' and entry.get('domain') == 'local' and entry['split'] != 'train':
            raise ValueError('Local adaptation sources must belong to training groups')
        for key in ['image', 'mask']:
            if not entry.get(key):
                if key == 'image':
                    raise ValueError('Source image filename is required')
                continue
            path = Path(entry[key])
            if not path.is_absolute():
                path = manifest_path.parent / path
            if not path.is_file():
                raise ValueError(f'Missing {key} for source {entry["id"]}')
            entry[key+'_path'] = path.resolve()
            entry[key+'_sha256'] = sha256(path)
        for mapping, key in [(group_splits, entry['group']), (image_splits, entry['image_sha256'])]:
            if key in mapping and mapping[key] != entry['split']:
                raise ValueError('Source group or identical image crosses train/val split')
            mapping[key] = entry['split']
        entry['dataset'] = entry.get('dataset', 'custom')
        records.append(entry)
    if not records:
        raise ValueError('Source manifest is empty')
    return records


def save_sample(output, sample_id, result, source, engine, save_qa):
    h, w = result['image'].shape[:2]
    split = source['split']
    image_rel = f'images/{split}/{sample_id}.jpg'
    label_rel = f'labels/{split}/{sample_id}.txt'
    rows = engine.yolo_rows(result['boxes_xyxy'], w, h)
    labels = ''.join('0 '+' '.join(f'{v:.9f}' for v in r[1:])+'\n' for r in rows)
    ok, encoded = cv2.imencode('.jpg', result['image'], [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise OSError('JPEG encoding failed')
    with (output/image_rel).open('xb') as handle:
        handle.write(encoded.tobytes())
    with (output/label_rel).open('x', encoding='utf-8') as handle:
        handle.write(labels)
    row = dict(result['metadata'], sample_id=sample_id, source_id=source['id'],
               source_group=source['group'], split=split, image_path=image_rel,
               label_path=label_rel, source_sha256=source['image_sha256'],
               boxes_xyxy=result['boxes_xyxy'], image_shape_hw=[h, w],
               image_sha256=sha256(output/image_rel), label_sha256=sha256(output/label_rel),
               stored_image_format='JPEG_quality95')
    if save_qa:
        folder = output/'qa'/sample_id
        folder.mkdir(parents=True)
        for key in ['image', 'clean_reference', 'nominal_source', 'label_mask', 'support_mask', 'effect_mask']:
            write_png(folder/(key+'.png'), result[key])
        write_json(folder/'metadata.json', row)
    with (output/'metadata.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    return row


def run(sources_file, output, count=100, profile='pretrain', seed=20261009,
        max_proposals=None, qa_count=6, cache_size=16, per_source_block=3):
    if profile not in ('pretrain', 'adaptation') or count < 1:
        raise ValueError('Expected a supported profile and positive sample count')
    if cache_size < 1 or per_source_block < 1 or qa_count < 0:
        raise ValueError('cache-size/per-source-block must be positive; qa-count cannot be negative')
    max_proposals = 4 * count if max_proposals is None else max_proposals
    if max_proposals < count:
        raise ValueError('max-proposals cannot be below the requested count')
    engine = importlib.import_module('.synthesis_'+profile, __package__)
    sources = load_sources(sources_file, profile)
    output = new_directory(output)
    for split in ('train', 'val'):
        (output/'images'/split).mkdir(parents=True, exist_ok=True)
        (output/'labels'/split).mkdir(parents=True, exist_ok=True)
    target = engine.quotas(count)
    accepted = Counter({k: 0 for k in target})
    failures, by_split, cache, unusable = Counter(), Counter(), OrderedDict(), {}
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(sources))
    source_cursor, within_source, proposal = 0, 0, 0
    config = {'profile': profile, 'generator_version': engine.VERSION,
              'source_manifest_sha256': sha256(sources_file), 'seed': int(seed),
              'count': count, 'strength_quotas': target, 'max_proposals': max_proposals,
              'cache_size': cache_size, 'per_source_block': per_source_block,
              'max_attempts_per_proposal': 32, 'qa_count': qa_count,
              'label_origin': 'algorithmic_synthetic', 'visual_review_required': True}
    write_json(output/'generation_config.json', config)
    while proposal < max_proposals and sum(accepted.values()) < count:
        if within_source >= per_source_block:
            within_source = 0
            source_cursor += 1
        if source_cursor >= len(order):
            order = rng.permutation(len(sources))
            source_cursor = 0
        source = sources[int(order[source_cursor])]
        within_source += 1
        needed = [k for k in target if accepted[k] < target[k]]
        strength = max(needed, key=lambda k: (target[k]-accepted[k])/max(target[k], 1))
        sample_seed = int(rng.integers(0, 2**31-1))
        mode = 'local_residual' if proposal % 2 == 0 else 'overlapping_views'
        proposal += 1
        try:
            sid = source['id']
            if sid in unusable:
                raise unusable[sid]
            if sid not in cache:
                try:
                    image = read_image(source['image_path'])
                    mask = read_image(source['mask_path'], grayscale=True) if source.get('mask_path') else None
                    info = {'safe_crop_bbox': source.get('crop_xyxy')}
                    if source.get('domain') == 'local':
                        info.update(patient_id=source['group'], split='train')
                    cache[sid] = engine.prepare_image(image, mask, info)
                except engine.SynthesisRejected as error:
                    unusable[sid] = error
                    raise
                while len(cache) > cache_size:
                    cache.popitem(last=False)
            cache.move_to_end(sid)
            result = engine.synthesize(cache[sid], sample_seed, strength, mode, max_attempts=32)
            sample_id = f'g{sum(accepted.values()):06d}_s{sample_seed}'
            save_sample(output, sample_id, result, source, engine,
                        sum(accepted.values()) < qa_count)
            accepted[strength] += 1
            by_split[source['split']] += 1
        except engine.SynthesisRejected as error:
            failures[error.reason] += 1
            with (output/'rejections.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps({'proposal': proposal, 'source_id': source['id'],
                    'seed': sample_seed, 'strength': strength, 'reason': error.reason,
                    'details': error.details, 'converted_to_negative': False})+'\n')
            if source['id'] in unusable:
                within_source = per_source_block
    completed = sum(accepted.values()) == count
    summary = {'profile': profile, 'completed': completed, 'accepted': sum(accepted.values()),
               'accepted_by_strength': dict(accepted), 'accepted_by_split': dict(by_split),
               'proposals': proposal, 'rejections': dict(failures),
               'quota_deficits': {k: target[k]-accepted[k] for k in target},
               'data_yaml_written': by_split['train'] > 0 and by_split['val'] > 0}
    if summary['data_yaml_written']:
        # Absolute output root is generated only on the user's machine. No
        # study dataset path is embedded in the distributed source package.
        yaml = 'path: '+json.dumps(output.resolve().as_posix())+'\ntrain: images/train\nval: images/val\nnames:\n  0: ghost\n'
        with (output/'data.yaml').open('x', encoding='utf-8') as handle:
            handle.write(yaml)
    write_json(output/'summary.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--profile', choices=['pretrain', 'adaptation'], default='pretrain')
    parser.add_argument('--seed', type=int, default=20261009)
    parser.add_argument('--max-proposals', type=int)
    parser.add_argument('--qa-count', type=int, default=6)
    parser.add_argument('--cache-size', type=int, default=16)
    parser.add_argument('--per-source-block', type=int, default=3)
    args = parser.parse_args(argv)
    try:
        summary = run(args.sources, args.output, args.count, args.profile, args.seed,
                      args.max_proposals, args.qa_count, args.cache_size, args.per_source_block)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f'Synthesis failed: {error}\n')
    print(json.dumps(summary))
    return 0 if summary['completed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
