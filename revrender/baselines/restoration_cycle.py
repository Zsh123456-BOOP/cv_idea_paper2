"""ICDAR 2021 ORIGINAL CycleGAN: official dense generators and PatchGANs.

Same source domains as paired methods, independently sampled. No ground-truth
stroke types, auxiliary recognizer, paired reconstruction or test data are used.
"""
import itertools

import numpy as np
import torch
import torch.nn.functional as F

from ..util import SEED, require, write_json
from .restoration import module_file
from .restoration_torch import build, image_tensor, predict_image


def train(e):
    a=e.args;torch.set_num_threads(4);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark=False
    root=a.upstream/a.model/'strikethrough_removal/src/network'
    init=module_file('official_cycle_init',root/'initialise.py').init_weights
    discriminator=module_file('official_cycle_discriminator',root/'discriminator.py').NLayerDiscriminator
    pool=module_file('official_cycle_pool',root/'image_pool.py').ImagePool
    # Follow official creation/initialization order: clean->struck, struck->clean, struck D, clean D.
    inverse,_,_=build(a.model,a.upstream/a.model)
    network,_,_=build(a.model,a.upstream/a.model)
    init(inverse);init(network)
    clean_d=discriminator(1).cuda();struck_d=discriminator(1).cuda()
    init(struck_d);init(clean_d)
    go=torch.optim.Adam(itertools.chain(inverse.parameters(),network.parameters()),lr=.0002,betas=(.5,.999))
    do=torch.optim.Adam(itertools.chain(struck_d.parameters(),clean_d.parameters()),lr=.0002,betas=(.5,.999))
    clean_pool=pool(20);struck_pool=pool(20)
    config={'architecture':'official DenseGenerator n_blocks1, BatchNorm; PatchGAN n_layers3',
            'variant':'ORIGINAL','loss':'LSGAN MSE + 10 L1 cycle each direction; identity lambda0',
            'paired_training':False,'type_labels':False,'auxiliary_recognizer':False,
            'lr':.0002,'betas':[.5,.999],'pool_size':20,'framework':torch.__version__,
            'parameters':sum(p.numel() for p in network.parameters()),
            'adaptation':'inverted grayscale native 256 tiles, batch4, independent domains from common training pool; seed42'}
    write_json(e.out/'model_config.json',config);e.log(config)

    def predict(x):return predict_image(network,None,x,a.model)

    def save(path):
        path.mkdir()
        torch.save({'model':network.state_dict(),'inverse':inverse.state_dict(),
                    'clean_d':clean_d.state_dict(),'struck_d':struck_d.state_dict(),
                    'generator_optimizer':go.state_dict(),'discriminator_optimizer':do.state_dict(),
                    'step':step,'config':config,'batch_rng':e.rng.bit_generator.state,
                    'unpaired_rng':e.unpaired_rng.bit_generator.state,
                    'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()},path/'weights.pt')
        restored=torch.load(path/'weights.pt',map_location='cpu',weights_only=False)['model']
        require(all(torch.equal(v.detach().cpu(),restored[k]) for k,v in network.state_dict().items()),'checkpoint reload mismatch')

    for step in range(1,a.steps+1):
        if step==1:before=next(network.parameters()).detach().clone()
        x,y=e.batch(unpaired=True);struck=image_tensor(x,a.model);clean=image_tensor(y,a.model)
        for m in (network,inverse,clean_d,struck_d):m.train()
        go.zero_grad(set_to_none=True)
        # Freezing parameter grads fixes the upstream attribute-only freeze (same G loss).
        clean_d.requires_grad_(False);struck_d.requires_grad_(False)
        fake_struck=inverse(clean);fake_clean=network(struck)
        cycle_clean=network(fake_struck);cycle_struck=inverse(fake_clean)
        dc=clean_d(fake_clean);ds=struck_d(fake_struck)
        gl=F.mse_loss(dc,torch.ones_like(dc))+F.mse_loss(ds,torch.ones_like(ds))
        gl=gl+10*F.l1_loss(cycle_clean,clean)+10*F.l1_loss(cycle_struck,struck)
        require(torch.isfinite(gl).item(),'nonfinite CycleGAN generator loss')
        gl.backward();go.step()
        clean_d.requires_grad_(True);struck_d.requires_grad_(True);do.zero_grad(set_to_none=True)
        rs=struck_d(struck);fs=struck_d(struck_pool.query(fake_struck.detach()))
        rc=clean_d(clean);fc=clean_d(clean_pool.query(fake_clean.detach()))
        dl=.5*(F.mse_loss(rs,torch.ones_like(rs))+F.mse_loss(fs,torch.zeros_like(fs))+
               F.mse_loss(rc,torch.ones_like(rc))+F.mse_loss(fc,torch.zeros_like(fc)))
        require(torch.isfinite(dl).item(),'nonfinite CycleGAN discriminator loss')
        dl.backward();do.step()
        if step==1:require(not torch.equal(before,next(network.parameters()).detach()),'CycleGAN did not update')
        if step==1 or step%100==0:e.log({'step':step,'generator_loss':float(gl.detach()),'discriminator_loss':float(dl.detach()),
                                      'peak_gpu_mib':torch.cuda.max_memory_allocated()/1024**2})
        if step%a.eval_every==0 or step==a.steps:
            e.evaluate(predict,step,save)
            if e.stop:break
    e.finish()
