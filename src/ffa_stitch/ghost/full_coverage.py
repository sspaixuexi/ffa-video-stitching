"""Conservative, single-original-frame replacement prototype. No anatomy claims.

Inputs are already projected onto the frozen panorama canvas. Rectangle unions
are editing candidates, not vessel or ghost segmentations. The exterior ring is
only a geometric proxy for a less affected boundary; it is not annotated normal.
"""
from __future__ import annotations
import time
import cv2
import numpy as np

VERSION = 'single-source-rectangle-pilot-0.1.1'
PARAMETERS = dict(context_px=5, transition_px=3, support_erosion_px=2,
                  minimum_full_sources=2, minimum_region_pixels=16,
                  minimum_ring_pixels=24, boundary_mae_max=18.,
                  boundary_q90_abs_max=45., boundary_gradient_mae_max=24.,
                  saturation_fraction_max=.15, contrast_min=3.,
                  boundary_mean_ratio_min=.65, boundary_mean_ratio_max=1.5,
                  r2_boundary_tie_gray=2.)

def gray(a):
    return cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)

def gradient(a):
    return cv2.magnitude(cv2.Sobel(a,cv2.CV_32F,1,0,ksize=3)/8.,
                         cv2.Sobel(a,cv2.CV_32F,0,1,ksize=3)/8.)

def integer_box(box,w,h):
    a=np.asarray(box,dtype=float)
    if a.shape!=(4,) or not np.isfinite(a).all():
        raise ValueError('Expected finite xyxy box')
    return [int(np.clip(np.floor(a[0]),0,w)),int(np.clip(np.floor(a[1]),0,h)),
            int(np.clip(np.ceil(a[2]),0,w)),int(np.clip(np.ceil(a[3]),0,h))]

def correct(baseline, donors, masks, boxes, donor_ids, policy='R1'):
    """Return image, masks, log and summary; RGB uint8. Never mutate inputs.

    masks are geometric support AND coarse tissue FOV, not vessel labels.
    R1 ranks guarded candidates by trimmed brightness, R2 by boundary match
    (within 2 gray levels) then contrast, gradient and brightness.
    Both branches use the same candidate eligibility and always read baseline.
    """
    start=time.perf_counter()
    if policy not in ('R1','R2'): raise ValueError(policy)
    h,w=baseline.shape[:2]
    if baseline.dtype!=np.uint8 or baseline.shape!=(h,w,3): raise ValueError('RGB uint8 expected')
    if donors.shape!=(len(donor_ids),h,w,3) or donors.dtype!=np.uint8: raise ValueError('donor dimensions')
    if masks.shape!=(len(donor_ids),h,w): raise ValueError('mask dimensions')
    if len(set(donor_ids))!=len(donor_ids): raise ValueError('duplicate donor IDs')
    output=baseline.copy(); union=np.zeros((h,w),np.uint8)
    iboxes=[]
    for b in boxes:
        x1,y1,x2,y2=integer_box(b,w,h); iboxes.append([x1,y1,x2,y2])
        if x2>x1 and y2>y1: union[y1:y2,x1:x2]=1
    # Adjacent/overlapping half-open rectangle pixels form a single operation.
    count,components=cv2.connectedComponents(union,connectivity=8)
    edit=np.zeros((h,w),bool); core_all=edit.copy(); log=[]
    baseg=gray(baseline); basegrad=gradient(baseg)
    p=PARAMETERS
    for k in range(1,count):
        region=components==k
        ys,xs=np.where(region); x1,x2=int(xs.min()),int(xs.max()+1); y1,y2=int(ys.min()),int(ys.max()+1)
        context_kernel=np.ones((2*p['context_px']+1,)*2,np.uint8)
        ring=cv2.dilate(region.astype(np.uint8),context_kernel).astype(bool)&~union.astype(bool)
        required=region|ring
        entry=dict(region_index=k-1,bbox=[x1,y1,x2,y2],region_pixels=int(region.sum()),
                   ring_pixels=int(ring.sum()),box_indices=[i for i,b in enumerate(iboxes) if b[2]>b[0] and b[3]>b[1] and region[b[1]:b[3],b[0]:b[2]].any()],
                   accepted=False,candidates=[])
        log.append(entry)
        if region.sum()<p['minimum_region_pixels'] or ring.sum()<p['minimum_ring_pixels']:
            entry['reason']='insufficient_region_or_exterior_ring'; continue
        full=[]
        for i in range(len(donor_ids)):
            support_kernel=np.ones((2*p['support_erosion_px']+1,)*2,np.uint8)
            safe=cv2.erode(masks[i].astype(np.uint8),support_kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
            covered=int(np.count_nonzero(safe&required)); coverage=covered/int(required.sum())
            c=dict(donor_index=i,donor_id=donor_ids[i],required_coverage=coverage,eligible=False,rejections=[])
            entry['candidates'].append(c)
            if covered!=int(required.sum()): c['rejections'].append('incomplete_region_or_ring_support'); continue
            full.append(c)
        entry['full_coverage_sources']=len(full)
        if len(full)<p['minimum_full_sources']:
            entry['reason']='fewer_than_two_fully_covering_sources'; continue
        bg=baseg[ring]
        for c in full:
            g=gray(donors[c['donor_index']]); v=g[region]; rg=g[ring]
            absdiff=np.abs(rg-bg); ns=v[v<250]
            if len(ns):
                lo,hi=np.percentile(ns,[10,90]); trimmed=ns[(ns>=lo)&(ns<=hi)]
                bright=float(trimmed.mean())
            else: bright=0.
            contrast=float(np.percentile(v,90)-np.percentile(v,10))
            grad=gradient(g)
            metrics=dict(boundary_mae=float(absdiff.mean()),boundary_q90_abs=float(np.percentile(absdiff,90)),
                         boundary_gradient_mae=float(np.abs(grad[ring]-basegrad[ring]).mean()),
                         boundary_mean_ratio=float((rg.mean()+1)/(bg.mean()+1)),
                         saturation_fraction=float((v>=250).mean()),contrast=contrast,
                         mean_gradient=float(grad[region].mean()),trimmed_brightness=bright)
            c['metrics']=metrics
            checks=[(metrics['boundary_mae']>p['boundary_mae_max'],'boundary_mae'),
                    (metrics['boundary_q90_abs']>p['boundary_q90_abs_max'],'boundary_q90'),
                    (metrics['boundary_gradient_mae']>p['boundary_gradient_mae_max'],'boundary_gradient'),
                    (not p['boundary_mean_ratio_min']<=metrics['boundary_mean_ratio']<=p['boundary_mean_ratio_max'],'boundary_exposure'),
                    (metrics['saturation_fraction']>p['saturation_fraction_max'],'core_saturation'),
                    (contrast<p['contrast_min'],'core_low_contrast')]
            c['rejections'] += [reason for failed,reason in checks if failed]
            c['eligible']=not c['rejections']
        good=[c for c in full if c['eligible']]
        if not good: entry['reason']='all_complete_donors_failed_quality_or_boundary'; continue
        if policy=='R1':
            selected=max(good,key=lambda c:(c['metrics']['trimmed_brightness'],-c['metrics']['boundary_mae'],-c['donor_index']))
        else:
            best=min(c['metrics']['boundary_mae'] for c in good)
            ties=[c for c in good if c['metrics']['boundary_mae']<=best+p['r2_boundary_tie_gray']]
            selected=max(ties,key=lambda c:(c['metrics']['contrast'],c['metrics']['mean_gradient'],c['metrics']['trimmed_brightness'],-c['donor_index']))
        # Distance is zero outside the candidate union; transition stays inside.
        padded=np.pad(region.astype(np.uint8),1)
        distance=cv2.distanceTransform(padded,cv2.DIST_L2,5)[1:-1,1:-1]
        alpha=np.minimum(distance/p['transition_px'],1.)
        core=region&(alpha>=1.)
        if not core.any():
            entry['reason']='no_single_source_core'; continue
        donor=donors[selected['donor_index']]
        blend=np.rint(baseline[region].astype(np.float32)*(1-alpha[region,None])+donor[region].astype(np.float32)*alpha[region,None]).clip(0,255).astype(np.uint8)
        output[region]=blend
        assert np.array_equal(output[core],donor[core]), 'single-source core contract'
        edit|=region; core_all|=core
        entry.update(accepted=True,reason='accepted',selected_donor_id=selected['donor_id'],
                     selected_donor_index=selected['donor_index'],selected_metrics=selected['metrics'],
                     core_pixels=int(core.sum()),transition_pixels=int((region&~core).sum()),
                     core_single_source_exact=True,
                     roi_mean_before=float(baseg[region].mean()),roi_mean_after=float(gray(output)[region].mean()),
                     changed_pixels=int(np.any(output[region]!=baseline[region],axis=1).sum()))
    unchanged=bool(np.array_equal(output[~edit],baseline[~edit]))
    assert unchanged
    return dict(image=output,edit_mask=edit,core_mask=core_all,log=log,
                summary=dict(version=VERSION,policy=policy,boxes=len(boxes),regions=len(log),
                             accepted_regions=sum(r['accepted'] for r in log),rejected_regions=sum(not r['accepted'] for r in log),
                             changed_pixels=int(np.any(output!=baseline,axis=2).sum()),
                             outside_edit_mask_unchanged=unchanged,
                             no_accepted_region_identity=bool(np.array_equal(output,baseline)) if not edit.any() else None,
                             runtime_seconds=time.perf_counter()-start))
