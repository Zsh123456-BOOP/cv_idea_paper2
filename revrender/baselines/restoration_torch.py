"""Load unchanged official restoration architectures and train task pairs."""
import copy
import math
import contextlib
import io
import logging
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F

from ..util import SEED, require, write_json
from .restoration import module_file, upstream_of


def package(name,path):
    m=types.ModuleType(name);m.__path__=[str(path)];sys.modules[name]=m
    return m


def build(model,root,architecture=None,stroke=False):
    if model=='NAFNet':
        # Avoid BasicSR's unrelated dataset/extension auto-imports. Architecture files
        # are unchanged, including their own LayerNorm and local pooling code.
        for name,sub in [('basicsr','basicsr'),('basicsr.models','basicsr/models'),('basicsr.models.archs','basicsr/models/archs')]:package(name,root/sub)
        utils=package('basicsr.utils',root/'basicsr/utils');utils.get_root_logger=lambda **kw:logging.getLogger('NAFNet')
        m=module_file('basicsr.models.archs.NAFNet_arch',root/'basicsr/models/archs/NAFNet_arch.py')
        options=dict(architecture or {})
        cls=getattr(m,options.pop('type','NAFNetLocal'))
        if not options:options=dict(img_channel=3,width=32,middle_blk_num=1,enc_blk_nums=[1,1,1,28],dec_blk_nums=[1,1,1,1])
        network=cls(**options).cuda()
        return network,None,None
    if model=='RevSep':
        from ..models.separation import RevSep
        return RevSep(build('NAFNet',root,architecture)[0],stroke=stroke).cuda(),None,None
    if model=='Restormer':
        m=module_file('official_restormer',root/'basicsr/models/archs/restormer_arch.py')
        return m.Restormer(**(architecture or {})).cuda(),None,None
    if model=='StainRestorer':
        m=module_file('official_stainrestorer',root/'models/StainRestorer.py')
        return m.StainRestorer(**(architecture or {})).cuda(),None,None
    if model in ('COR-Generator','Strikethrough-CycleGAN'):
        file=root/'src/_models.py' if model=='COR-Generator' else root/'strikethrough_removal/src/network/generators.py'
        m=module_file('official_dense_'+model.replace('-','_'),file)
        return m.DenseGenerator(1,1,n_blocks=1).cuda(),None,None
    require(model=='DocDiff','unknown restoration architecture')
    package('src',root/'src');package('model',root/'model');package('schedule',root/'schedule')
    m=module_file('model.DocDiff',root/'model/DocDiff.py')
    from src.sobel import Laplacian
    from schedule.schedule import Schedule
    from schedule.diffusionSample import GaussianDiffusion
    network=m.DocDiff(input_channels=6,output_channels=3,n_channels=32,ch_mults=[1,2,3,4],n_blocks=1).cuda()
    diffusion=GaussianDiffusion(network.denoiser,100,Schedule('linear',100)).cuda()
    return network,diffusion,Laplacian().cuda()


def docdiff_loss(network,diffusion,lap,x,y):
    t=torch.randint(0,100,(len(x),),device=x.device)
    initial,residual,_,_=network(y,x,t,diffusion)
    truth=y-initial
    ddpm=2*F.mse_loss(lap(residual),lap(truth))+F.mse_loss(residual,truth)
    pixel=F.mse_loss(initial,y)+2*F.mse_loss(initial-lap(initial),y-lap(y))
    return ddpm+.5*pixel


def image_tensor(x,model):
    t=torch.from_numpy(np.ascontiguousarray(x[:,None])).cuda()
    if model in ('COR-Generator','Strikethrough-CycleGAN'):return 1-t
    return t.repeat(1,3,1,1)


def final_image(output,model):
    # The release infer.py passes a list into metrics; train.py correctly uses [0].
    return output[0] if model=='StainRestorer' else output


def stain_loss(outputs,target):
    from torchmetrics.functional.image import structural_similarity_index_measure
    # Both stages are supervised, exactly as in the author's train.py.
    return sum(F.mse_loss(value,target)+.2*(1-structural_similarity_index_measure(value,target))
               for value in outputs)


@torch.no_grad()
def predict_image(network,diffusion,x,model):
    network.eval()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(SEED);xx=image_tensor(x,model)
        if diffusion is not None:
            initial=network.init_predictor(xx,torch.zeros(len(xx),dtype=torch.long,device=xx.device))
            with contextlib.redirect_stdout(io.StringIO()):
                out=initial+diffusion(torch.randn_like(initial),initial,pre_ori='True')
        else:
            # Native validation may batch several tiles. Keep the same per-image
            # memory bound as StainRestorer training without changing its output.
            if model=='StainRestorer':out=torch.cat([final_image(network(im[None]),model) for im in xx])
            else:out=network(xx)
    if model in ('COR-Generator','Strikethrough-CycleGAN'):out=1-out
    return out.mean(1).clamp(0,1).cpu().numpy()


def train(e):
    a=e.args;require(torch.cuda.is_available(),'CUDA required')
    torch.set_num_threads(4);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark=False
    network,diffusion,lap=build(a.model,upstream_of(a.upstream,a.model),stroke=getattr(a,'stroke',False))
    if a.init:network.load_state_dict(torch.load(a.init/'weights.pt',map_location='cuda',weights_only=False)['model'])
    scale=a.learning_rate_scale
    is_doc=a.model=='DocDiff';is_cor=a.model=='COR-Generator';is_rest=a.model=='Restormer'
    is_stain=a.model=='StainRestorer';is_sep=a.model=='RevSep'
    if is_cor:opt=torch.optim.Adam(network.parameters(),lr=1e-3*scale,betas=(.9,.999))
    elif is_stain:opt=torch.optim.AdamW(network.parameters(),lr=2e-4*scale,betas=(.9,.999),weight_decay=.01,eps=1e-8)
    else:
        opt=torch.optim.AdamW(network.parameters(),lr=scale*(1e-4 if is_doc else 3e-4 if is_rest else 1e-3),
            weight_decay=1e-4 if is_doc or is_rest else 1e-3,
            betas=(.9,.999) if is_doc or is_rest else (.9,.9))
    scheduler=None
    if a.model in ('NAFNet','RevSep'):scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.steps,eta_min=1e-7*scale)
    if is_stain:scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.steps,eta_min=1e-6*scale)
    if is_rest:
        # Preserve the official flat/cosine fractions, scale duration to declared task budget.
        flat=max(1,round(a.steps*92000/300000))
        scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda s: 1. if s<=flat else
            1e-6/3e-4+(1-1e-6/3e-4)*.5*(1+math.cos(math.pi*(s-flat)/max(1,a.steps-flat))))
    ema=copy.deepcopy(network).eval() if is_doc and a.ema else None
    ema_diffusion=None
    if ema is not None:
        from schedule.diffusionSample import GaussianDiffusion
        from schedule.schedule import Schedule
        ema_diffusion=GaussianDiffusion(ema.denoiser,100,Schedule('linear',100)).cuda()
        ema.requires_grad_(False)
    config={'parameters':sum(p.numel() for p in network.parameters()),'framework':torch.__version__,
        'architecture':a.model+' unchanged official architecture',
        'loss':('official PRE_ORI L2 high/low frequency, beta50/T100' if is_doc else
                'BCE on inverted images' if is_cor else 'L1' if is_rest else
                'both stages: MSE + 0.2*(1-SSIM), author torchmetrics defaults per image' if is_stain else 'official PSNR loss'),
        'diffusion_steps':100 if is_doc else None,'ema':ema is not None,'precision':'float32',
        'ema_rule':'official .9999 parameter average every 100 updates after step 2000; initial model copy; raw and EMA validation candidates' if ema is not None else None,
        'adaptation':'256 native tiles; batch4; seed42; no rotation; task-specific budget; Restormer fixed patches and scaled flat/cosine schedule',
        'initialization':str(a.init) if a.init else 'official random initialization','learning_rate_scale':scale}
    if e.w is not None:
        require(a.model=='NAFNet','per-pixel weights are honoured by NAFNet only')
        config['loss']+=", weighted per pixel by the data's weights (0 where the pseudo-label teachers disagree)"
    if is_sep:
        config.update(architecture='RevSep (models.separation): the unchanged official NAFNet body on grey, hint and zeros; a 1x1 erase/cross head on its last decoder features',
            loss=('PSNR loss of the composed output and of H1, erase BCE+Dice, cross BCE' if a.heads else 'PSNR loss of H1'),
            heads=a.heads,hint=a.hint,erase_weight=a.erase_weight,cross_weight=a.cross_weight,hint_reach=a.hint_reach,hint_drop=a.hint_drop,
            reader=str(a.reader) if a.reader else None,reading_weight=a.reading_weight,stroke=a.stroke,hint_from=a.hint_from,
            stroke_weights=[a.centre_weight,a.width_weight,a.stroke_weight] if a.stroke else None)
    if is_stain:
        import torchmetrics
        config.update(initialization=str(a.init) if a.init else 'random; no author checkpoint',torchmetrics=torchmetrics.__version__,
            microbatch=1,adaptation='native256; effective batch4 via single-image gradients; image-wise SSIM matches author batch1 config; cosine scaled to task update budget; seed42; no rotation')
    write_json(e.out/'model_config.json',config);e.log(config)
    active=network

    def predict(x,hint=None):
        if is_sep:return separate(active,x,np.zeros_like(x) if hint is None else hint,a.heads,hint_from=a.hint_from)
        return predict_image(active,ema_diffusion if active is ema else diffusion,x,a.model)

    def save(path):
        path.mkdir()
        torch.save({'model':active.state_dict(),'raw_model':network.state_dict(),
            'ema_model':ema.state_dict() if ema is not None else None,'optimizer':opt.state_dict(),
            'scheduler':scheduler.state_dict() if scheduler else None,'step':step,'config':config,
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'batch_rng':e.rng.bit_generator.state},path/'weights.pt')
        restored=torch.load(path/'weights.pt',map_location='cpu',weights_only=False)['model']
        require(all(torch.equal(v.detach().cpu(),restored[k]) for k,v in active.state_dict().items()),'checkpoint round-trip mismatch')

    if is_sep:
        from ..models.separation import separate, separation_loss
        reading=None
        if a.reader and a.reading_weight:
            from ..reader_ctc import ctc, load
            reader,alphabet=load(a.reader)
            def reading_of(ids):     # the one-tile human regions of the batch, each read on its window of the output
                chosen=[(k,e.rows[i]) for k,i in enumerate(ids) if 'text' in e.rows[i]]
                if not chosen:return lambda out:None
                def term(out):
                    crops=[out[k,0,y:y+h,x:x+w] for k,r in chosen for x,y,w,h in [r['window']]]
                    return a.reading_weight*ctc(reader,crops,[r['text'] for _,r in chosen],alphabet)
                return term
            reading=reading_of
    terms={}
    for step in range(1,a.steps+1):
        if step==1:before=next(network.parameters()).detach().clone()
        if is_sep:
            x,y,hint,erase,cross,ids,strokes=e.labelled_batch();network.train();opt.zero_grad(set_to_none=True)
            t=lambda v:torch.from_numpy(np.ascontiguousarray(v[:,None])).cuda()
            loss,terms=separation_loss(network,t(x),t(y),t(hint),t(erase),t(cross),a.heads,a.erase_weight,a.cross_weight,
                                       reading(ids) if reading else None,tuple(map(t,strokes)) if strokes else None,a.hint_from,
                                       (a.centre_weight,a.width_weight,a.stroke_weight) if a.stroke else (1.,1.,1.))
            require(torch.isfinite(loss).item(),'nonfinite training loss');value=float(loss.detach());loss.backward()
        else:
            x,y,w=e.batch(weights=True);network.train();opt.zero_grad(set_to_none=True)
            # These architectures have no batch norm. Accumulate an effective batch4.
            size=1 if is_rest or is_stain else len(x);value=0.
            for start in range(0,len(x),size):
                xx=image_tensor(x[start:start+size],a.model);yy=image_tensor(y[start:start+size],a.model)
                if is_doc:loss=docdiff_loss(network,diffusion,lap,xx,yy)
                elif is_cor:loss=F.binary_cross_entropy(network(xx),yy)
                elif is_rest:loss=F.l1_loss(network(xx),yy)
                elif is_stain:loss=stain_loss(network(xx),yy)
                elif e.w is not None:   # NAFNet on weighted data (data.pseudo_regions)
                    from ..models.separation import psnr_loss
                    loss=psnr_loss(network(xx),yy,torch.from_numpy(np.ascontiguousarray(w[start:start+size,None])).cuda().expand_as(yy))
                else:loss=10/np.log(10)*torch.log(((network(xx)-yy)**2).mean((1,2,3))+1e-8).mean()
                require(torch.isfinite(loss).item(),'nonfinite training loss')
                loss=loss*(len(xx)/len(x));value+=float(loss.detach());loss.backward()
        if a.model in ('NAFNet','RevSep'):torch.nn.utils.clip_grad_norm_(network.parameters(),.01)
        if is_rest:torch.nn.utils.clip_grad_norm_(network.parameters(),.01)
        opt.step()
        if step==1:require(not torch.equal(before,next(network.parameters()).detach()),'model did not update')
        if ema is not None and step>2000 and step%100==0:
            with torch.no_grad():
                for target,current in zip(ema.parameters(),network.parameters()):target.mul_(.9999).add_(current,alpha=.0001)
        if scheduler:scheduler.step()
        if step==1 or step%100==0:e.log({'step':step,'loss':value,'lr':opt.param_groups[0]['lr'],
            'peak_gpu_mib':torch.cuda.max_memory_allocated()/1024**2,**({'terms':terms} if is_sep else {})})
        if step%a.eval_every==0 or step==a.steps:
            active=network
            if ema is not None:
                e.evaluate(predict,step,save,variant='raw',final_candidate=False)
                active=ema;e.evaluate(predict,step,save,variant='ema')
            else:e.evaluate(predict,step,save)
            if e.stop:break
    e.finish()
