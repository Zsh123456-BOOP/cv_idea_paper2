"""ICDAR 2021 CycleGANs for strikethrough removal: official dense generators and PatchGANs.

Strikethrough-CycleGAN is the release's ORIGINAL: same source domains as paired methods, independently sampled; no stroke types,
auxiliary recognizer, paired reconstruction or test data.  Strikethrough-CycleGAN-A is its ATTR_GUIDED configuration (serverConfig.cfg:
experiment FEATURE_RECOG, featuretype CHANNEL, cnn_lambda 0.1, disc_feature True): the clean-to-struck generator and the struck critic
also take the struck sample's attribute as one-hot channels, and the L1 between an auxiliary DenseNet-121's features of the real and
the generated struck images is added at 0.1.  The attribute here is the annotation mode of the region (``ATTRIBUTES``: erase the
strike / keep the writing, plan of 2026-11-18), not the authors' seven stroke types; the DenseNet is trained first on the training
tiles' struck side to tell the two, as the release's strikethrough_classification trains it (cross entropy, Adam 2e-4 (.5, .999),
batch 64, 30 epochs, the epoch of the best validation F1, inverted grey), on our 256 tiles and the validation writers' windows.  The
struck-to-clean generator restores from the image alone.
"""
import itertools

import numpy as np
import torch
import torch.nn.functional as F

from ..util import SEED, require, write_json
from .restoration import module_file
from .restoration_torch import build, image_tensor, predict_image

ATTRIBUTES = ('erase', 'keep')
RECOGNISER = {'epochs': 30, 'batch': 64, 'lr': 2e-4, 'betas': (.5, .999), 'cnn_lambda': .1}


def macro_f1(truth, guess):
    return float(np.mean([2*((truth == c) & (guess == c)).sum()/max(1, (truth == c).sum()+(guess == c).sum()) for c in range(len(ATTRIBUTES))]))


def stroke_recogniser(e):
    """The auxiliary DenseNet-121 (one grey channel in, one class per attribute), trained on the training tiles that hold a strike and
    kept at the epoch of its best macro F1 on the validation windows' tiles that hold one: (its feature extractor, its record)."""
    import types
    from torchvision import models
    from ..data.restoration_pairs import canvas_tiles
    train = np.array([i for i in e.train if not e.rows[i]['clean']]); labels = np.array([ATTRIBUTES.index(e.rows[i]['mode']) for i in train])
    val_x, val_y = [], []
    for i, r in enumerate(e.native_rows):
        for tile, mask in zip(canvas_tiles(e.native[f'image_{i}'], 255), canvas_tiles(e.native[f'mask_{i}'], 0)):
            if mask.any(): val_x.append(tile); val_y.append(ATTRIBUTES.index(r['mode']))
    val_x, val_y = np.stack(val_x).astype(np.float32)/255, np.array(val_y)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
        model = models.densenet121(weights=None, num_classes=len(ATTRIBUTES), memory_efficient=True)   # memory_efficient: same values
        first = model.features.conv0
        model.features.conv0 = torch.nn.Conv2d(1, first.out_channels, kernel_size=first.kernel_size, stride=first.stride, padding=first.padding,
                                               dilation=first.dilation, groups=first.groups, bias=first.bias, padding_mode=first.padding_mode)
        model = model.cuda(); opt = torch.optim.Adam(model.parameters(), lr=RECOGNISER['lr'], betas=RECOGNISER['betas'])
        best, kept, curve = -1., None, []
        for epoch in range(1, RECOGNISER['epochs']+1):
            model.train(); order = rng.permutation(len(train)); losses = []
            for start in range(0, len(order), RECOGNISER['batch']):
                pick = order[start:start+RECOGNISER['batch']]
                x = 1-torch.from_numpy(e.x[train[pick]].astype(np.float32)/255)[:, None].cuda()
                loss = F.cross_entropy(model(x), torch.from_numpy(labels[pick]).cuda())
                opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss))
            model.eval()
            with torch.no_grad():
                guess = np.concatenate([model(1-torch.from_numpy(val_x[s:s+RECOGNISER['batch']])[:, None].cuda()).argmax(1).cpu().numpy()
                                        for s in range(0, len(val_x), RECOGNISER['batch'])])
            f1 = macro_f1(val_y, guess); curve.append({'epoch': epoch, 'loss': float(np.mean(losses)), 'val_macro_f1': f1, 'val_accuracy': float((guess == val_y).mean())})
            e.log({'recogniser': curve[-1]})
            if f1 > best: best, kept = f1, {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(kept); model.eval(); model.requires_grad_(False)
    torch.save(kept, e.out/'stroke_recogniser.pt')

    def features(self, x):   # the release's FEATURE_RECOG replacement: the classification layer dropped
        return torch.flatten(F.adaptive_avg_pool2d(F.relu(self.features(x)), (1, 1)), 1)
    model.forward = types.MethodType(features, model)
    record = {'attributes': list(ATTRIBUTES), 'train_tiles': {c: int((labels == k).sum()) for k, c in enumerate(ATTRIBUTES)},
              'validation_tiles': {c: int((val_y == k).sum()) for k, c in enumerate(ATTRIBUTES)}, 'best_val_macro_f1': best,
              'kept_epoch': 1+int(np.argmax([c['val_macro_f1'] for c in curve])), 'curve': curve, **RECOGNISER}
    write_json(e.out/'stroke_recogniser.json', record)
    return model, record


def train(e):
    a=e.args;torch.set_num_threads(4);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark=False
    attributed=a.model=='Strikethrough-CycleGAN-A';base='Strikethrough-CycleGAN';source=a.upstream/base
    root=source/'strikethrough_removal/src/network'
    init=module_file('official_cycle_init',root/'initialise.py').init_weights
    discriminator=module_file('official_cycle_discriminator',root/'discriminator.py').NLayerDiscriminator
    pool=module_file('official_cycle_pool',root/'image_pool.py').ImagePool
    extra=len(ATTRIBUTES) if attributed else 0   # one-hot attribute channels on the clean-to-struck generator and the struck critic
    # Follow official creation/initialization order: clean->struck, struck->clean, struck D, clean D.
    if attributed:
        inverse=module_file('official_dense_Strikethrough_CycleGAN',root/'generators.py').DenseGenerator(1+extra,1,n_blocks=1).cuda()
    else:inverse,_,_=build(base,source)
    network,_,_=build(base,source)
    init(inverse);init(network)
    clean_d=discriminator(1).cuda();struck_d=discriminator(1+extra).cuda()
    init(struck_d);init(clean_d)
    if a.init:   # all four nets of the chosen version: a CycleGAN fine-tunes as it trained
        state=torch.load(a.init/'weights.pt',map_location='cuda',weights_only=False)
        for net,key in ((network,'model'),(inverse,'inverse'),(clean_d,'clean_d'),(struck_d,'struck_d')):net.load_state_dict(state[key])
    lr=.0002*a.learning_rate_scale
    go=torch.optim.Adam(itertools.chain(inverse.parameters(),network.parameters()),lr=lr,betas=(.5,.999))
    do=torch.optim.Adam(itertools.chain(struck_d.parameters(),clean_d.parameters()),lr=lr,betas=(.5,.999))
    clean_pool=pool(20);struck_pool=pool(20)
    recogniser,recognised=stroke_recogniser(e) if attributed else (None,None)
    config={'architecture':'official DenseGenerator n_blocks1, BatchNorm; PatchGAN n_layers3',
            'variant':'ATTR_GUIDED (attribute = annotation mode, 属性为标注模式)' if attributed else 'ORIGINAL',
            'loss':'LSGAN MSE + 10 L1 cycle each direction; identity lambda0'+(' + 0.1 L1 of the auxiliary DenseNet features' if attributed else ''),
            'paired_training':False,'type_labels':list(ATTRIBUTES) if attributed else False,'auxiliary_recognizer':recognised and {k:v for k,v in recognised.items() if k!='curve'},
            'lr':lr,'betas':[.5,.999],'initialization':str(a.init) if a.init else 'official init_weights','pool_size':20,'framework':torch.__version__,
            'parameters':sum(p.numel() for p in network.parameters()),
            'adaptation':'inverted grayscale native 256 tiles, batch4, independent domains from common training pool; seed42'}
    write_json(e.out/'model_config.json',config);e.log(config)

    def predict(x):return predict_image(network,None,x,base)

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
        x,y=e.batch(unpaired=True);struck=image_tensor(x,base);clean=image_tensor(y,base)
        if attributed:   # the struck samples' attributes, one-hot over the tile (dataset.py FeatureType.CHANNEL)
            kinds=torch.tensor([ATTRIBUTES.index(e.rows[i]['mode']) for i in e.last_ids],device='cuda')
            feature=F.one_hot(kinds,len(ATTRIBUTES)).float()[:,:,None,None].expand(-1,-1,*struck.shape[2:])
            with_feature=lambda t:torch.cat((t,feature),1)
        else:with_feature=lambda t:t
        for m in (network,inverse,clean_d,struck_d):m.train()
        go.zero_grad(set_to_none=True)
        # Freezing parameter grads fixes the upstream attribute-only freeze (same G loss).
        clean_d.requires_grad_(False);struck_d.requires_grad_(False)
        fake_struck=inverse(with_feature(clean));fake_clean=network(struck)
        cycle_clean=network(fake_struck);cycle_struck=inverse(with_feature(fake_clean))
        dc=clean_d(fake_clean);ds=struck_d(with_feature(fake_struck))
        gl=F.mse_loss(dc,torch.ones_like(dc))+F.mse_loss(ds,torch.ones_like(ds))
        gl=gl+10*F.l1_loss(cycle_clean,clean)+10*F.l1_loss(cycle_struck,struck)
        if attributed:gl=gl+RECOGNISER['cnn_lambda']*F.l1_loss(recogniser(fake_struck),recogniser(struck))
        require(torch.isfinite(gl).item(),'nonfinite CycleGAN generator loss')
        gl.backward();go.step()
        clean_d.requires_grad_(True);struck_d.requires_grad_(True);do.zero_grad(set_to_none=True)
        rs=struck_d(with_feature(struck));fs=struck_d(struck_pool.query(with_feature(fake_struck).detach()))
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
