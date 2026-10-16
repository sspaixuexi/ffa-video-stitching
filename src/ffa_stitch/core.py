"""Adaptive keyframe selection and chronological two-stage FFA stitching.

Preserved: first-three-frame attrition denominator, componentwise MNN tolerance,
target-endpoint re-match filtering, similarity RANSAC, raw-frame ROI partition,
chronological two-stage stitching, fixed first-frame canvas and max fusion.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict
import time
import re
import cv2
import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union
from .matching import image_hash

# Keep classical image operators on the CPU to avoid an extra OpenCL context
# competing with PyTorch CUDA.
cv2.ocl.setUseOpenCL(False)


@dataclass
class Config:
    selection: str='adaptive'
    fixed_step: int=2
    min_tracks: int=15
    track_ratio: float=0.2
    track_tolerance: float=10.0
    rematch_tolerance: float=20.0
    refine_matches: bool=True
    partition: bool=True
    transform: str='similarity'
    canvas: str='fixed'
    fusion: str='max'
    include_last: bool=False
    repair_partition_reset: bool=False
    max_canvas_pixels: int=20_000_000
    max_global_match_side: int=0
    seed: int=20260916
    motion_fraction: float=0.10
    motion_max_gap: int=10


class StageTimes:
    def __init__(self):
        self.values=defaultdict(float)

    @contextmanager
    def stage(self,name):
        t=time.perf_counter()
        try:
            yield
        finally:
            self.values[name] += time.perf_counter()-t


def roi_and_mask(gray):
    blurred=cv2.GaussianBlur(gray,(5,5),0)
    binary=cv2.threshold(blurred,30,255,cv2.THRESH_BINARY)[1]
    binary=cv2.dilate(binary,cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)),iterations=2)
    contours=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    contours=[c for c in contours if cv2.contourArea(c)>500]
    if not contours:
        return None,np.zeros(gray.shape,np.uint8)
    c=max(contours,key=cv2.contourArea)
    x,y,w,h=cv2.boundingRect(c)
    rect=(max(x-10,0),max(y-10,0),min(x+w+10,gray.shape[1]),min(y+h+10,gray.shape[0]))
    mask=np.zeros_like(gray);cv2.drawContours(mask,[c],-1,255,cv2.FILLED)
    return rect,mask


def prepare_frames(images):
    frames=[]
    for i,raw in enumerate(images):
        h,w=raw.shape
        size=(max(32,w//32*32),max(32,h//32*32))
        im=cv2.resize(raw,size) if (w,h)!=size else raw.copy()
        roi,mask=roi_and_mask(raw)
        mask=cv2.resize(mask,size,interpolation=cv2.INTER_NEAREST)
        frames.append(dict(image=im,key=image_hash(im),roi=roi,mask=mask,index=i,
                           original_shape=(h,w),original_to_work=np.diag([size[0]/w,size[1]/h,1.])))
    return frames


def load_sequence(folder):
    """Read 8-bit grayscale frames in natural filename order from a directory.

    Inputs are already cropped FFA frames, as used by the stitching pipeline.
    The function does not infer a clinical crop from a full surgical recording.
    """
    folder=Path(folder)
    if not folder.is_dir():
        raise ValueError('Input frames directory does not exist: '+str(folder))
    extensions={'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}
    paths=[p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in extensions]
    def natural_key(path):
        return [(0,int(s)) if s.isdigit() else (1,s.lower()) for s in re.split(r'(\d+)',path.name)]
    paths.sort(key=natural_key)
    images=[cv2.imdecode(np.fromfile(p,dtype=np.uint8),cv2.IMREAD_GRAYSCALE) for p in paths]
    if not images or any(im is None for im in images):
        raise ValueError('Missing or unreadable input frames: '+str(folder))
    return prepare_frames(images),paths


def mutual_indices(a,b,tolerance,block=256):
    """Exact float32 NumPy-distance MNN, chunked to avoid the dense N*M*2 tensor.

    First-index tie breaking and the original componentwise <= test are retained.
    """
    a=np.asarray(a,np.float32);b=np.asarray(b,np.float32)
    if len(a)==0 or len(b)==0:
        return np.empty(0,int),np.empty(0,int)
    ab=np.empty(len(a),int)
    ba=np.zeros(len(b),int)
    best=np.full(len(b),np.inf,dtype=np.float32)
    for start in range(0,len(a),block):
        dist=np.linalg.norm(a[start:start+block,None,:]-b[None,:,:],axis=-1)
        ab[start:start+len(dist)]=np.argmin(dist,axis=1)
        loc=np.argmin(dist,axis=0)
        val=dist[loc,np.arange(len(b))]
        improve=val<best
        ba[improve]=start+loc[improve];best[improve]=val[improve]
    ids=np.arange(len(a))
    valid=(ba[ab]==ids)&(np.max(np.abs(a-b[ab]),axis=1)<=tolerance)
    return ids[valid],ab[valid]


def match_frames(engine,frames,a,b):
    fa,fb=frames[a],frames[b]
    return engine.match(fa['image'],fb['image'],fa['key'],fb['key'])


def select_keyframes(frames,engine,cfg):
    n=len(frames)
    if n==0: raise ValueError('Empty sequence')
    targets={};track_log=[]
    if cfg.selection=='fixed':
        kfs=list(range(0,n,cfg.fixed_step))
    elif cfg.selection=='motion':
        kfs=motion_keyframes(frames,cfg)
    elif cfg.selection=='adaptive':
        kfs=[0]
        if n>=2:
            _,current,_=match_frames(engine,frames,0,1)
            first_check=True;initial=0
            for i in range(1,n-1):
                pa,pb,_=match_frames(engine,frames,i,i+1)
                ia,ib=mutual_indices(current,pa,cfg.track_tolerance)
                count=len(ia)
                if first_check:
                    initial=count;first_check=False
                trigger=count<cfg.min_tracks or (count/max(initial,1)<=cfg.track_ratio)
                track_log.append(dict(frame=i,next_frame=i+1,tracks=count,initial=initial,trigger=trigger))
                if trigger:
                    # Current survivors refer to frame i, before accepting i+1.
                    # The first iteration uses the matched endpoint in the current frame.
                    target=pa[ib] if i==1 else current.copy()
                    targets[(kfs[-1],i)]=target
                    kfs.append(i)
                    current=pb.copy();first_check=True
                else:
                    current=pb[ib]
    else:
        raise ValueError(cfg.selection)
    if cfg.include_last and kfs[-1]!=n-1:
        kfs.append(n-1)
    return kfs,targets,track_log


def motion_keyframes(frames,cfg):
    kfs=[0];previous=None;points=None;anchor=None
    scale=512/max(frames[0]['image'].shape)
    for i,f in enumerate(frames):
        im=cv2.resize(f['image'],None,fx=scale,fy=scale)
        if previous is None or previous.shape!=im.shape or points is None or len(points)<6:
            if i and kfs[-1]!=i: kfs.append(i)
            points=cv2.goodFeaturesToTrack(im,500,0.01,8)
            anchor=None if points is None else points.copy()
        else:
            nxt,status,_=cv2.calcOpticalFlowPyrLK(previous,im,points,None,winSize=(21,21),maxLevel=3)
            good=np.zeros(len(points),bool) if status is None else status.ravel()>0
            surviving=int(good.sum())
            displacement=float(np.median(np.linalg.norm(nxt[good,0]-anchor[good,0],axis=1))) if surviving else np.inf
            trigger=(surviving<20 or surviving/max(len(points),1)<0.5 or
                     displacement>cfg.motion_fraction*min(im.shape) or i-kfs[-1]>=cfg.motion_max_gap)
            if trigger:
                kfs.append(i);points=cv2.goodFeaturesToTrack(im,500,0.01,8)
                anchor=None if points is None else points.copy()
            else:
                points=nxt[good];anchor=anchor[good]
        previous=im
    return kfs


class NotebookPartition:
    def __init__(self):
        self.older=None;self.last=None

    def add(self,rect):
        if rect is None: return True
        new=box(*rect)
        if self.last is None:
            self.last=new;return True
        if new.within(self.last): return True
        if self.older is not None and not new.intersection(self.older).is_empty:
            self.older=None;self.last=None;return False
        total=self.last if self.older is None else unary_union([self.older,self.last])
        self.older=total.difference(new);self.last=new
        return True


def estimate(a,b,model='similarity',seed=20260916):
    if len(a)<3: return None,0
    a=np.asarray(a,np.float32);b=np.asarray(b,np.float32)
    cv2.setRNGSeed(seed)
    if model=='similarity':
        h,m=cv2.estimateAffinePartial2D(b,a,method=cv2.RANSAC,ransacReprojThreshold=5)
    elif model=='affine':
        h,m=cv2.estimateAffine2D(b,a,method=cv2.RANSAC,ransacReprojThreshold=5)
    elif model=='homography':
        if len(a)<4: return None,0
        h,m=cv2.findHomography(b,a,cv2.RANSAC,5)
    else: raise ValueError(model)
    if h is None or not np.isfinite(h).all(): return None,0
    if h.shape==(2,3): h=np.vstack([h,[0.,0.,1.]])
    if abs(np.linalg.det(h))<1e-10: return None,0
    return h,int(m.sum()) if m is not None else 0


def transform_points(h,pts):
    v=np.column_stack([pts,np.ones(len(pts))])@h.T
    return v[:,:2]/v[:,2,None]


def warp(im,h,size,nearest=False):
    flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    if np.allclose(h[2],[0,0,1],atol=1e-12):
        return cv2.warpAffine(im,h[:2],size,flags=flags)
    return cv2.warpPerspective(im,h,size,flags=flags)


def bounds(images,transforms,cfg):
    if cfg.canvas=='fixed':
        return (images[0].shape[1],images[0].shape[0]),np.eye(3)
    corners=[]
    for im,h in zip(images,transforms):
        height,width=im.shape
        p=transform_points(h,np.array([[0,0],[width,0],[width,height],[0,height]],float))
        if not np.isfinite(p).all(): raise ValueError('nonfinite_canvas')
        corners.extend(p)
    corners=np.asarray(corners)
    lo=np.floor(corners.min(0));hi=np.ceil(corners.max(0))
    w,h=(hi-lo).astype(int)
    if w<1 or h<1 or w*h>cfg.max_canvas_pixels:
        raise ValueError('canvas_limit:'+str((int(w),int(h))))
    offset=np.array([[1,0,-lo[0]],[0,1,-lo[1]],[0,0,1]],float)
    return (int(w),int(h)),offset


def blend(a,ma,b,mb,mode):
    if mode=='max': return np.maximum(a,b),np.maximum(ma,mb)
    if mode=='masked_max':
        return np.maximum(np.where(ma>0,a,0),np.where(mb>0,b,0)).astype(np.uint8),np.maximum(ma,mb)
    if not np.any(ma): return b.copy(),mb.copy()
    if not np.any(mb): return a.copy(),ma.copy()
    if mode=='feather':
        da=cv2.distanceTransform((ma>0).astype(np.uint8),cv2.DIST_L2,3)
        db=cv2.distanceTransform((mb>0).astype(np.uint8),cv2.DIST_L2,3)
        alpha=db/(da+db+1e-8)
        result=np.clip(a*(1-alpha)+b*alpha,0,255).astype(np.uint8)
    elif mode in ('multiband','graphcut'):
        colors=[cv2.cvtColor(x,cv2.COLOR_GRAY2BGR) for x in (a,b)]
        masks=[ma.copy(),mb.copy()]
        if mode=='graphcut':
            umasks=[cv2.UMat(x) for x in masks]
            seam=cv2.detail_GraphCutSeamFinder('COST_COLOR_GRAD')
            seam.find([x.astype(np.float32) for x in colors],[(0,0),(0,0)],umasks)
            masks=[x.get() for x in umasks]
        blender=cv2.detail_MultiBandBlender()
        blender.setNumBands(4)
        h,w=a.shape;blender.prepare((0,0,w,h))
        for color,mask in zip(colors,masks):
            blender.feed(color.astype(np.int16),mask,(0,0))
        color,_=blender.blend(None,None)
        result=cv2.cvtColor(np.clip(color,0,255).astype(np.uint8),cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(mode)
    result[(ma>0)&(mb==0)]=a[(ma>0)&(mb==0)]
    result[(mb>0)&(ma==0)]=b[(mb>0)&(ma==0)]
    result[(ma==0)&(mb==0)]=0
    return result,np.maximum(ma,mb)


def render(frames,kfs,transforms,cfg):
    images=[frames[k]['image'] for k in kfs]
    size,off=bounds(images,transforms,cfg)
    canvas=np.zeros(size[::-1],np.uint8);mask=np.zeros_like(canvas)
    actual={};source_geometry={}
    for k,h in zip(kfs,transforms):
        h=off@h;actual[k]=h
        source_geometry[k]=dict(H_work_to_local=h.copy(),H_local_to_canvas=np.eye(3),
                                local_canvas_size=size,
                                projection_route=[dict(matrix=h.copy(),canvas_size_wh=size)])
        f=frames[k]
        canvas,mask=blend(canvas,mask,warp(f['image'],h,size),warp(f['mask'],h,size,True),cfg.fusion)
    return dict(image=canvas,mask=mask,transforms=actual,keyframes=kfs.copy(),
                source_geometry=source_geometry)


def stitch(frames,engine,cfg):
    timer=StageTimes();started=time.perf_counter();before=engine.counters()
    with timer.stage('keyframe_tracking'):
        kfs,targets,tracks=select_keyframes(frames,engine,cfg)
    checker=NotebookPartition();checker.add(frames[kfs[0]]['roi'])
    local_specs=[];current=[kfs[0]];transforms=[np.eye(3)]
    edges=[];events=[]
    for a,b in zip(kfs[:-1],kfs[1:]):
        with timer.stage('keyframe_rematch_registration_partition'):
            pa,pb,confidence=match_frames(engine,frames,a,b)
            raw_count=len(pa)
            if cfg.refine_matches and cfg.selection=='adaptive' and b>a+1 and (a,b) in targets:
                ids,_=mutual_indices(pb,targets[(a,b)],cfg.rematch_tolerance)
                pa,pb=pa[ids],pb[ids]
            h,inliers=estimate(pa,pb,cfg.transform,cfg.seed)
            cut=cfg.partition and not checker.add(frames[b]['roi'])
            row=dict(a=a,b=b,direct_matches=raw_count,used_matches=len(pa),inliers=inliers,
                     H_b_to_a=h,partition_cut=bool(cut),registration_failed=h is None)
            edges.append(row)
            if cut or h is None:
                local_specs.append((current,transforms))
                current=[b];transforms=[np.eye(3)]
                events.append(dict(frame=b,event='partition' if cut else 'registration_failure'))
                if cfg.repair_partition_reset or (h is None and not cut):
                    checker=NotebookPartition();checker.add(frames[b]['roi'])
            else:
                current.append(b);transforms.append(transforms[-1]@h)
    local_specs.append((current,transforms))
    local=[]
    with timer.stage('local_warp_blending'):
        for group,hs in local_specs:
            try:
                local.append(render(frames,group,hs,cfg))
            except ValueError as e:
                events.append(dict(event='render_failure',keyframes=group,reason=str(e)))
                # Explicitly retain failed frames as disconnected pieces. No
                # identity warp is silently accepted as successful registration.
                for k in group: local.append(render(frames,[k],[np.eye(3)],cfg))
    pieces=[local[0]];merge_log=[]
    for incoming in local[1:]:
        base=pieces[0]
        with timer.stage('global_rematch_registration'):
            aa,bb=base['image'],incoming['image']
            scales=[];small=[]
            for im in (aa,bb):
                s=min(1.,cfg.max_global_match_side/max(im.shape)) if cfg.max_global_match_side else 1.
                small.append(cv2.resize(im,None,fx=s,fy=s) if s<1 else im)
                scales.append(np.diag([small[-1].shape[1]/im.shape[1],small[-1].shape[0]/im.shape[0],1.]))
            pa,pb,_=engine.match(*small)
            h,inl=estimate(pa,pb,cfg.transform,cfg.seed)
            if h is not None: h=np.linalg.inv(scales[0])@h@scales[1]
        if h is None:
            pieces.append(incoming)
            merge_log.append(dict(incoming=incoming['keyframes'],matches=len(pa),inliers=inl,success=False,reason='registration'))
            continue
        try:
            with timer.stage('global_warp_blending'):
                size,off=bounds([aa,bb],[np.eye(3),h],cfg)
                ca=warp(aa,off,size);ma=warp(base['mask'],off,size,True)
                cb=warp(bb,off@h,size);mb=warp(incoming['mask'],off@h,size,True)
                im,mask=blend(ca,ma,cb,mb,cfg.fusion)
                hs={k:off@v for k,v in base['transforms'].items()}
                hs.update({k:off@h@v for k,v in incoming['transforms'].items()})
                source_geometry={k:dict(v,H_local_to_canvas=off@v['H_local_to_canvas'],
                                       projection_route=v['projection_route']+[dict(matrix=off.copy(),canvas_size_wh=size)])
                                 for k,v in base['source_geometry'].items()}
                source_geometry.update({k:dict(v,H_local_to_canvas=off@h@v['H_local_to_canvas'],
                                              projection_route=v['projection_route']+[dict(matrix=off@h,canvas_size_wh=size)])
                                        for k,v in incoming['source_geometry'].items()})
                pieces[0]=dict(image=im,mask=mask,transforms=hs,keyframes=base['keyframes']+incoming['keyframes'],
                               source_geometry=source_geometry)
            merge_log.append(dict(incoming=incoming['keyframes'],matches=len(pa),inliers=inl,success=True,H_b_to_a=h))
        except ValueError as e:
            pieces.append(incoming)
            merge_log.append(dict(incoming=incoming['keyframes'],matches=len(pa),inliers=inl,success=False,reason=str(e)))
    after=engine.counters()
    return dict(config=asdict(cfg),n_inputs=len(frames),keyframes=kfs,tracks=tracks,edges=edges,
                local_groups=[g for g,_ in local_specs],n_local_panoramas=len(local),pieces=pieces,
                merges=merge_log,events=events,stage_times=dict(timer.values),pipeline_s=time.perf_counter()-started,
                matcher_counters={k:after[k]-before[k] for k in before})
