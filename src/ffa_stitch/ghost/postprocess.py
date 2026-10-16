"""Optional final-panorama postprocessing from traceable original-frame donors.

No detector, reference image or manual truth enters this module. B uses partial
coverage and brightness ranking. C uses exterior context for bounded translation,
photometric fitting and source ranking. Rejection retains the input exactly.
"""
from __future__ import annotations
import time
import cv2
import numpy as np

VERSION='source-postprocess-0.2.0'
PARAMETERS=dict(support_erosion=2,minimum_pixels=128,minimum_fraction=.20,
                feather=12,context_gap=24,context_radius=56,minimum_context=128,
                max_translation=12,translation_coarse_step=6,ncc_gain_for_shift=.03,
                max_candidates=12,minimum_ncc=.20,gain_min=.85,gain_max=1.20,
                offset_bound=15.,boundary_mae_max=18.,boundary_q90_max=45.,
                core_contrast_min=3.,saturation_max=.15,
                highpass_sigma=2.,minimum_structure_difference=1.,minimum_q95_change=6.)

def gray(a):return cv2.cvtColor(a,cv2.COLOR_RGB2GRAY).astype(np.float32)
def highpass(g):return g-cv2.GaussianBlur(g,(0,0),PARAMETERS['highpass_sigma'])
def erode(a,r):return cv2.erode(a.astype(np.uint8),np.ones((2*r+1,)*2,np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
def dilate(a,r):return cv2.dilate(a.astype(np.uint8),np.ones((2*r+1,)*2,np.uint8))>0
def move(a,dx,dy,mask=False):
    if dx==0 and dy==0:return a
    h,w=a.shape[:2]
    result=cv2.warpAffine(a.astype(np.uint8) if mask else a,np.array([[1,0,dx],[0,1,dy]],np.float32),(w,h),flags=cv2.INTER_NEAREST if mask else cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    return result>0 if mask else result
def trimmean(v):
    a,b=np.percentile(v,[10,90]);return float(v[(v>=a)&(v<=b)].mean())
def ncc(a,b,m):
    if m.sum()<PARAMETERS['minimum_context']:return None
    x=a[m].astype(np.float64);y=b[m].astype(np.float64);x-=x.mean();y-=y.mean()
    norm=np.sqrt(np.sum(x*x)*np.sum(y*y))
    return float(np.sum(x*y)/norm) if norm>1e-8 else None
def photometry(donor,target,mask):
    x=donor[mask].astype(np.float64);y=target[mask].astype(np.float64)
    step=max(1,len(x)//30000);x=x[::step];y=y[::step]
    use=np.ones(len(x),bool);gain=1.;offset=0.
    if np.std(x)<2:return gain,offset
    for _ in range(3):
        xx=x[use];yy=y[use]
        gain=float(np.sum((xx-xx.mean())*(yy-yy.mean()))/(np.sum((xx-xx.mean())**2)+1e-9))
        gain=float(np.clip(gain,PARAMETERS['gain_min'],PARAMETERS['gain_max']))
        offset=float(np.clip(np.median(yy-gain*xx),-PARAMETERS['offset_bound'],PARAMETERS['offset_bound']))
        error=np.abs(y-(gain*x+offset));use=error<=np.percentile(error,80)
    return gain,offset
def keep_large(mask):
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
    keep=np.zeros(n,bool)
    if n>1:keep[1:]=stats[1:,cv2.CC_STAT_AREA]>=PARAMETERS['minimum_pixels']
    return keep[labels]

def postprocess(baseline,donors,content_masks,support_masks,boxes,donor_ids,mode='C'):
    start=time.perf_counter();p=PARAMETERS
    if mode not in ['B','C']:raise ValueError(mode)
    h,w=baseline.shape[:2];N=len(donor_ids)
    if baseline.dtype!=np.uint8 or baseline.shape!=(h,w,3):raise ValueError('RGB uint8 input required')
    if donors.shape!=(N,h,w,3) or donors.dtype!=np.uint8:raise ValueError('donor shape/type')
    if content_masks.shape!=(N,h,w) or support_masks.shape!=(N,h,w):raise ValueError('mask shape')
    if len(set(donor_ids))!=N:raise ValueError('duplicate source IDs')
    union=np.zeros((h,w),bool);integer_boxes=[]
    for box in boxes:
        b=np.asarray(box,float)
        if b.shape!=(4,) or not np.isfinite(b).all():raise ValueError('finite xyxy boxes required')
        x1,y1,x2,y2=[int(v) for v in [max(0,np.floor(b[0])),max(0,np.floor(b[1])),min(w,np.ceil(b[2])),min(h,np.ceil(b[3]))]]
        if x2>x1 and y2>y1:union[y1:y2,x1:x2]=True
        integer_boxes.append([x1,y1,x2,y2])
    output=baseline.copy();edit=np.zeros((h,w),bool);cores=edit.copy();logs=[]
    n,labels=cv2.connectedComponents(union.astype(np.uint8),8)
    if n==1 or N<2:
        logs=[dict(region_index=k-1,accepted=False,reason='fewer_than_two_input_sources') for k in range(1,n)]
        return dict(image=output,edit_mask=edit,core_mask=cores,log=logs,summary=dict(version=VERSION,mode=mode,boxes=len(boxes),regions=len(logs),accepted_regions=0,rejected_regions=len(logs),changed_pixels=0,edited_pixels=0,core_pixels=0,outside_edit_mask_exact=True,runtime_seconds=time.perf_counter()-start))
    safe=np.stack([erode((content_masks[i]>0)&(support_masks[i]>0),p['support_erosion']) for i in range(N)])
    counts=safe.sum(axis=0);baseg=gray(baseline);basehp=highpass(baseg)
    for label in range(1,n):
        region=labels==label;ys,xs=np.where(region);bbox=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]
        valid_region=region&(counts>0);initial_dual=region&(counts>=2)
        log=dict(region_index=label-1,bbox=bbox,candidate_pixels=int(region.sum()),valid_content_pixels=int(valid_region.sum()),
                 initial_dual_pixels=int(initial_dual.sum()),accepted=False,candidates=[])
        logs.append(log)
        minimum=max(p['minimum_pixels'],p['minimum_fraction']*int(valid_region.sum()))
        if initial_dual.sum()<minimum:
            log['reason']='insufficient_initial_dual_content';continue
        pad=p['context_radius']+p['max_translation']+4
        x0=max(0,bbox[0]-pad);y0=max(0,bbox[1]-pad);xend=min(w,bbox[2]+pad);yend=min(h,bbox[3]+pad)
        sl=np.s_[y0:yend,x0:xend]
        reg=region[sl];du=initial_dual[sl];u=union[sl];bg=baseg[sl];bhp=basehp[sl];base=baseline[sl]
        ctx=dilate(reg,p['context_radius'])&~dilate(u,p['context_gap'])&(counts[sl]>0)
        pre=[]
        for i in range(N):
            cov=du&safe[i][sl]
            if cov.sum()>=minimum:
                g=gray(donors[i][sl]);pre.append((int(cov.sum()),trimmean(g[cov]),i))
        pre.sort(key=lambda r:(-r[0],-r[1],r[2]));pre=pre[:p['max_candidates']]
        accepted=[]
        for _,__,i in pre:
            donor=donors[i][sl];g=gray(donor);hp=highpass(g);si=safe[i][sl]
            c=dict(donor_index=i,donor_id=donor_ids[i],eligible=False,rejections=[],dx=0,dy=0,gain=1.,offset=0.)
            log['candidates'].append(c)
            initial_ctx=ctx&si;initial_corr=ncc(bhp,hp,initial_ctx);c['initial_context_ncc']=initial_corr
            moved_si=si;moved_donor=donor;corr=initial_corr
            if mode=='C':
                if initial_corr is None:
                    c['rejections'].append('insufficient_textured_exterior_context');continue
                # Fixed context pixels for every shift avoids score gains from
                # silently discarding difficult points while searching.
                searchctx=ctx&erode(si,p['max_translation'])
                basecorr=ncc(bhp,hp,searchctx)
                if basecorr is not None:
                    candidates=[]
                    for dy in range(-p['max_translation'],p['max_translation']+1,p['translation_coarse_step']):
                        for dx in range(-p['max_translation'],p['max_translation']+1,p['translation_coarse_step']):
                            cc=ncc(bhp,move(hp,dx,dy),searchctx)
                            if cc is not None:candidates.append((cc,-abs(dx)-abs(dy),-dy,-dx,dx,dy))
                    coarse=max(candidates);cx,cy=coarse[-2:]
                    for dy in range(max(-p['max_translation'],cy-3),min(p['max_translation'],cy+3)+1):
                        for dx in range(max(-p['max_translation'],cx-3),min(p['max_translation'],cx+3)+1):
                            cc=ncc(bhp,move(hp,dx,dy),searchctx)
                            if cc is not None:candidates.append((cc,-abs(dx)-abs(dy),-dy,-dx,dx,dy))
                    best=max(candidates)
                    if best[0]>=basecorr+p['ncc_gain_for_shift']:
                        c['dx'],c['dy']=best[-2:];moved_si=move(si,c['dx'],c['dy'],True);moved_donor=move(donor,c['dx'],c['dy'])
                    c['search_context_pixels']=int(searchctx.sum());c['search_ncc_before']=basecorr;c['search_ncc_best']=best[0]
                g=gray(moved_donor);hp=highpass(g);fitctx=ctx&moved_si
                corr=ncc(bhp,hp,fitctx)
                if corr is None or corr<p['minimum_ncc']:
                    c['rejections'].append('low_exterior_structure_agreement');continue
                c['gain'],c['offset']=photometry(g,bg,fitctx)
            proposal=np.rint(moved_donor.astype(np.float32)*c['gain']+c['offset']).clip(0,255).astype(np.uint8)
            pg=gray(proposal)
            # Initial overlap is immutable. An alignment cannot restore a
            # region where source availability was absent in the inputs.
            other=(counts[sl]-si.astype(np.int64))>=1
            candidate=keep_large(du&moved_si&other)
            if candidate.sum()<minimum:
                c['rejections'].append('insufficient_partial_coverage');continue
            distance=cv2.distanceTransform(np.pad(candidate.astype(np.uint8),1),cv2.DIST_L2,5)[1:-1,1:-1]
            core=candidate&(distance>=p['feather']);band=candidate&(distance<=4)
            if core.sum()<p['minimum_pixels'] or band.sum()<24:
                c['rejections'].append('insufficient_single_source_core');continue
            boundary=np.abs(pg[band]-bg[band]);diff=np.abs(pg[core]-bg[core]);structured=np.abs(highpass(pg)[core]-bhp[core])
            contrast=float(np.percentile(pg[core],90)-np.percentile(pg[core],10))
            metrics=dict(edited_pixels=int(candidate.sum()),edited_fraction_of_box=float(candidate.sum()/region.sum()),
                         edited_fraction_of_valid_content=float(candidate.sum()/valid_region.sum()),context_ncc=corr,
                         boundary_mae=float(boundary.mean()),boundary_q90=float(np.percentile(boundary,90)),
                         contrast=contrast,saturation=float((pg[core]>=250).mean()),
                         structural_difference=float(structured.mean()),q95_change=float(np.percentile(diff,95)),
                         trimmed_brightness=trimmean(pg[core]),core_pixels=int(core.sum()))
            c['metrics']=metrics
            gates=[(metrics['boundary_mae']>p['boundary_mae_max'],'boundary_mae'),
                   (metrics['boundary_q90']>p['boundary_q90_max'],'boundary_q90'),
                   (contrast<p['core_contrast_min'],'low_contrast'),
                   (metrics['saturation']>p['saturation_max'],'saturation'),
                   (metrics['structural_difference']<p['minimum_structure_difference'] or metrics['q95_change']<p['minimum_q95_change'],'no_clear_structural_change')]
            c['rejections'] += [name for bad,name in gates if bad]
            c['eligible']=not c['rejections']
            if c['eligible']:
                score=(metrics['trimmed_brightness'],-metrics['boundary_mae'],-i) if mode=='B' else (float(corr),-metrics['boundary_mae'],metrics['edited_fraction_of_valid_content'],-i)
                accepted.append((score,c,proposal,candidate,core,distance))
        if not accepted:
            log['reason']='no_eligible_source';continue
        _,selected,proposal,candidate,core,distance=max(accepted,key=lambda a:a[0])
        alpha=np.minimum(distance/p['feather'],1.)
        outview=output[sl]
        outview[candidate]=np.rint(base[candidate].astype(np.float32)*(1-alpha[candidate,None])+proposal[candidate].astype(np.float32)*alpha[candidate,None]).clip(0,255).astype(np.uint8)
        assert np.array_equal(outview[core],proposal[core])
        edit[sl]|=candidate;cores[sl]|=core
        log.update(accepted=True,reason='accepted_partial_postprocessing',selected_donor_index=selected['donor_index'],
                   selected_donor_id=selected['donor_id'],selected_dx=selected['dx'],selected_dy=selected['dy'],
                   selected_gain=selected['gain'],selected_offset=selected['offset'],metrics=selected['metrics'],
                   unchanged_candidate_pixels=int(region.sum()-candidate.sum()),crop_xyxy=[x0,y0,xend,yend],
                   core_equals_transformed_photometric_source=True)
    assert np.array_equal(output[~edit],baseline[~edit])
    return dict(image=output,edit_mask=edit,core_mask=cores,log=logs,
                summary=dict(version=VERSION,mode=mode,boxes=len(boxes),regions=len(logs),accepted_regions=sum(r['accepted'] for r in logs),
                             rejected_regions=sum(not r['accepted'] for r in logs),changed_pixels=int(np.any(output!=baseline,axis=2).sum()),
                             edited_pixels=int(edit.sum()),core_pixels=int(cores.sum()),outside_edit_mask_exact=True,runtime_seconds=time.perf_counter()-start))
