"""Load unchanged official NAFNet / DocDiff architectures and train task pairs."""
import contextlib
import io
import logging
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F

from ..util import SEED, require, write_json
from .restoration import module_file


def package(name,path):
    m=types.ModuleType(name);m.__path__=[str(path)];sys.modules[name]=m
    return m


def build(model,root):
    if model=='NAFNet':
        # Avoid BasicSR's unrelated dataset/extension auto-imports. Architecture files
        # are unchanged, including their own LayerNorm and local pooling code.
        for name,sub in [('basicsr','basicsr'),('basicsr.models','basicsr/models'),('basicsr.models.archs','basicsr/models/archs')]:package(name,root/sub)
        utils=package('basicsr.utils',root/'basicsr/utils');utils.get_root_logger=lambda **kw:logging.getLogger('NAFNet')
        m=module_file('basicsr.models.archs.NAFNet_arch',root/'basicsr/models/archs/NAFNet_arch.py')
        network=m.NAFNetLocal(img_channel=3,width=32,middle_blk_num=1,enc_blk_nums=[1,1,1,28],dec_blk_nums=[1,1,1,1]).cuda()
        return network,None,None
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


def train(e):
    a=e.args;require(torch.cuda.is_available(),'CUDA required')
    torch.set_num_threads(4);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark=False
    network,diffusion,lap=build(a.model,a.upstream/a.model)
    is_doc=a.model=='DocDiff'
    opt=torch.optim.AdamW(network.parameters(),lr=1e-4 if is_doc else 1e-3,
        weight_decay=1e-4 if is_doc else 1e-3,betas=(.9,.999) if is_doc else (.9,.9))
    scheduler=None if is_doc else torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.steps,eta_min=1e-7)
    config={'parameters':sum(p.numel() for p in network.parameters()),'framework':torch.__version__,
        'architecture':'official DocDiff 32/[1,2,3,4]/1' if is_doc else 'official NAFNetLocal width32 enc[1,1,1,28]',
        'loss':'official PRE_ORI L2 high/low frequency, beta50/T100' if is_doc else 'official PSNR loss',
        'diffusion_steps':100 if is_doc else None,'ema':False,'precision':'float32',
        'adaptation':'short-budget pilot uses raw DocDiff weights, no EMA; native 256 tiling, no rotation of handwriting'}
    write_json(e.out/'model_config.json',config);e.log(config)

    def tensor(x):return torch.from_numpy(np.ascontiguousarray(x[:,None])).cuda().repeat(1,3,1,1)

    @torch.no_grad()
    def predict(x):
        network.eval()
        # Evaluation must not consume the training RNG; fixed diffusion noise.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(SEED);xx=tensor(x)
            if is_doc:
                initial=network.init_predictor(xx,torch.zeros(len(xx),dtype=torch.long,device=xx.device))
                with contextlib.redirect_stdout(io.StringIO()):
                    out=initial+diffusion(torch.randn_like(initial),initial,pre_ori='True')
            else:out=network(xx)
        return out.mean(1).clamp(0,1).cpu().numpy()

    def save(path):
        path.mkdir()
        torch.save({'model':network.state_dict(),'optimizer':opt.state_dict(),
            'scheduler':scheduler.state_dict() if scheduler else None,'step':step,'config':config,
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'batch_rng':e.rng.bit_generator.state},path/'weights.pt')
        # Round-trip the checkpoint and check every tensor, without resetting the optimizer.
        restored=torch.load(path/'weights.pt',map_location='cpu',weights_only=False)['model']
        require(all(torch.equal(v.detach().cpu(),restored[k]) for k,v in network.state_dict().items()),'checkpoint round-trip mismatch')

    for step in range(1,a.steps+1):
        x,y=e.batch();network.train();opt.zero_grad(set_to_none=True);xx,yy=tensor(x),tensor(y)
        if is_doc:loss=docdiff_loss(network,diffusion,lap,xx,yy)
        else:loss=10/np.log(10)*torch.log(((network(xx)-yy)**2).mean((1,2,3))+1e-8).mean()
        require(torch.isfinite(loss).item(),'nonfinite training loss')
        loss.backward()
        # NAFNet's official trainer clips norm to .01; DocDiff does not.
        if not is_doc:torch.nn.utils.clip_grad_norm_(network.parameters(),.01)
        opt.step()
        if scheduler:scheduler.step()
        if step==1 or step%100==0:e.log({'step':step,'loss':float(loss.detach()),'lr':opt.param_groups[0]['lr']})
        if step%a.eval_every==0 or step==a.steps:e.evaluate(predict,step,save)
    e.finish()
