"""Feature matchers with an optional content-addressed correspondence cache."""
from pathlib import Path
from copy import deepcopy
import hashlib
import json
import sys
import time
import types
import zipfile

import cv2
import numpy as np


def image_hash(image):
    return hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()


class MatchEngine:
    """Match grayscale images in their supplied pixel coordinates.

    EfficientLoFTR uses an external source checkout and checkpoint. SIFT and
    ORB support CPU-only examples without downloading learned model weights.
    The default EfficientLoFTR configuration uses full precision on CUDA.
    """

    def __init__(self, name='eloftr', cache_read=True, cache_write=True, coarse=0.7,
                 *, cache_dir=None, eloftr_repo=None, weights=None, device='cuda'):
        self.name = name
        self.cache_read = cache_read and cache_dir is not None
        self.cache_write = cache_write and cache_dir is not None
        self.coarse = coarse
        self.device = device
        self.model = None
        self.memory = {}
        self.calls = self.hits_memory = self.hits_disk = self.computed = 0
        self.compute_s = self.disk_s = 0.0
        self.corrupt_cache_files = 0
        started = time.perf_counter()
        self.metadata = {'matcher': name, 'working_coordinates': True, 'coarse': coarse}
        if name == 'eloftr':
            if eloftr_repo is None or weights is None:
                raise ValueError('EfficientLoFTR requires eloftr_repo and weights paths')
            self._load_eloftr(Path(eloftr_repo), Path(weights))
        elif name == 'sift':
            self.model = cv2.SIFT_create(nfeatures=4096)
            self.metadata.update(nfeatures=4096, ratio=0.8, device='cpu')
        elif name == 'orb':
            self.model = cv2.ORB_create(nfeatures=4096, fastThreshold=10)
            self.metadata.update(nfeatures=4096, ratio=0.8, fastThreshold=10, device='cpu')
        else:
            raise ValueError('Unsupported matcher: '+name)
        self.metadata.update(opencv=cv2.__version__)
        cache_id = hashlib.sha256(json.dumps(self.metadata,sort_keys=True).encode()).hexdigest()[:16]
        self.cache_dir = Path(cache_dir or '.ffa_match_cache') / (name+'_'+cache_id)
        if self.cache_read or self.cache_write:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.load_s = time.perf_counter()-started

    def _load_eloftr(self, repo, weights):
        import torch
        import loguru
        if not (repo/'src/loftr').is_dir():
            raise ValueError('Expected an EfficientLoFTR checkout containing src/loftr')
        if not weights.is_file():
            raise ValueError('EfficientLoFTR checkpoint does not exist')
        self.device = torch.device(self.device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; select --device cpu explicitly for CPU inference')
        # Inference imports a training-only logging decorator. Supply its small
        # interface without installing the complete training framework.
        if not hasattr(loguru, '_Logger'):
            loguru._Logger = type(loguru.logger)
        if 'pytorch_lightning' not in sys.modules:
            pl = types.ModuleType('pytorch_lightning'); pl.__path__ = []
            utilities = types.ModuleType('pytorch_lightning.utilities')
            def rank_zero_only(fn):
                return fn
            rank_zero_only.rank = 0
            utilities.rank_zero_only = rank_zero_only
            sys.modules['pytorch_lightning'] = pl
            sys.modules['pytorch_lightning.utilities'] = utilities
            # The upstream checkpoint uses this class as a callbacks-state key;
            # the model state dictionary is loaded unchanged.
            callbacks = types.ModuleType('pytorch_lightning.callbacks'); callbacks.__path__ = []
            checkpoint_module = types.ModuleType('pytorch_lightning.callbacks.model_checkpoint')
            checkpoint_module.ModelCheckpoint = type('ModelCheckpoint',(),{})
            sys.modules['pytorch_lightning.callbacks'] = callbacks
            sys.modules['pytorch_lightning.callbacks.model_checkpoint'] = checkpoint_module
        sys.path.insert(0, str(repo.resolve()))
        from src.loftr import LoFTR, full_default_cfg, reparameter
        cfg = deepcopy(full_default_cfg)
        cfg['match_coarse']['thr'] = self.coarse
        self.model = LoFTR(config=cfg)
        checkpoint = torch.load(weights,map_location='cpu',weights_only=False)
        self.model.load_state_dict(checkpoint['state_dict'])
        del checkpoint
        self.model = reparameter(self.model).eval().to(self.device)
        loguru.logger.remove()
        loguru.logger.add(sys.stderr, level='ERROR')
        source = b''.join(p.read_bytes() for p in sorted((repo/'src/loftr').rglob('*.py')))
        self.metadata.update(torch=torch.__version__, cuda=torch.version.cuda,
                             device=str(self.device), precision='fp32',
                             matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                             cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                             weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                             model_source_sha256=hashlib.sha256(source).hexdigest(), config=cfg)
        if self.device.type == 'cuda':
            self.metadata['device_name'] = torch.cuda.get_device_name(self.device)

    def clear_memory(self):
        self.memory.clear()

    def counters(self):
        return dict(calls=self.calls,memory_hits=self.hits_memory,disk_hits=self.hits_disk,
                    computed=self.computed,compute_s=self.compute_s,disk_s=self.disk_s,
                    corrupt_cache_files=self.corrupt_cache_files)

    def match(self, a, b, key_a=None, key_b=None):
        self.calls += 1
        ka, kb = key_a or image_hash(a), key_b or image_hash(b)
        key = (ka,kb)
        if key in self.memory:
            self.hits_memory += 1
            return self.memory[key]
        fn = self.cache_dir/(hashlib.sha256((ka+'|'+kb).encode()).hexdigest()+'.npz')
        value = None
        if self.cache_read and fn.exists():
            t = time.perf_counter()
            try:
                with np.load(fn,allow_pickle=False) as z:
                    value = (z['a'],z['b'],z['confidence'])
                self.hits_disk += 1
            except (zipfile.BadZipFile,ValueError,EOFError,OSError):
                fn.rename(fn.with_suffix('.corrupt_'+str(time.time_ns())))
                self.corrupt_cache_files += 1
            self.disk_s += time.perf_counter()-t
        if value is None:
            t = time.perf_counter()
            value = self._compute(a,b)
            self.compute_s += time.perf_counter()-t
            self.computed += 1
            if self.cache_write:
                t = time.perf_counter()
                temporary = fn.with_suffix('.part')
                with temporary.open('wb') as f:
                    np.savez_compressed(f,a=value[0],b=value[1],confidence=value[2])
                temporary.replace(fn)
                self.disk_s += time.perf_counter()-t
        self.memory[key] = value
        return value

    def _compute(self,a,b):
        if self.name in ('sift','orb'):
            pa,da = self.model.detectAndCompute(a,None)
            pb,db = self.model.detectAndCompute(b,None)
            if da is None or db is None or len(da)<2 or len(db)<2:
                return np.empty((0,2),np.float32),np.empty((0,2),np.float32),np.empty(0,np.float32)
            norm = cv2.NORM_L2 if self.name=='sift' else cv2.NORM_HAMMING
            pairs = cv2.BFMatcher(norm).knnMatch(da,db,k=2)
            good = [v[0] for v in pairs if len(v)==2 and v[0].distance<0.8*v[1].distance]
            return (np.asarray([pa[m.queryIdx].pt for m in good],np.float32).reshape(-1,2),
                    np.asarray([pb[m.trainIdx].pt for m in good],np.float32).reshape(-1,2),
                    np.asarray([1/(1+m.distance) for m in good],np.float32))
        import torch
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        with torch.inference_mode():
            tensors = []; scales = []
            for im in (a,b):
                h,w = im.shape
                h32,w32 = max(32,h//32*32),max(32,w//32*32)
                x = im if (h,w)==(h32,w32) else cv2.resize(im,(w32,h32))
                tensors.append(torch.from_numpy(x)[None,None].float().to(self.device)/255.)
                scales.append(np.array([w/w32,h/h32],np.float32))
            batch = {'image0':tensors[0],'image1':tensors[1]}
            self.model(batch)
            aa = batch['mkpts0_f'].cpu().numpy()*scales[0]
            bb = batch['mkpts1_f'].cpu().numpy()*scales[1]
            cc = batch['mconf'].cpu().numpy()
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        return np.asarray(aa,np.float32),np.asarray(bb,np.float32),np.asarray(cc,np.float32)
