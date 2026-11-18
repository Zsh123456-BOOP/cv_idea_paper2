"""VCGAN (IJDAR 2024; shivii/VCGAN @56cb974), unpaired, with the one change that gives its Top-k term a gradient ("修梯度").

The released train.py makes the Top-k term ``torch.mean((torch.tensor(diff_k)).clone().detach())``: no gradient reaches the generators
through it, and with the cycle L1 commented out and the identity weight 0, nothing else ties a translation to its input.  Here, and
only here, the term is ``torch.mean(diff_k)`` (``GRADIENT_FIX``).  Everything else is the release's: ResNet generators (9 blocks) and
PatchGAN critics (sigmoid) on 3 channels in [-1, 1], least-squares adversarial terms, the Top-k cycle term (the largest tenth of the
absolute differences) weighted 10 each way, identity 0, Adam 1e-5 (.5, .999), batch 1, mixed precision, the critics stepped before
the generators, created in the release's order (critic A, critic B, generator B, generator A).  A is the erased (clean) domain, B the
struck one; generator A (B to A) is the restorer.  Adaptation: grey tiles repeated to 3 channels; the two domains are the training
tiles' struck and erased sides, drawn independently (as Strikethrough-CycleGAN here); the release's resize to 256 is our 256 tiles.
"""
import torch
import torch.nn.functional as F

from ..util import SEED, require, write_json
from .restoration import module_file, upstream_of
from .restoration_torch import image_tensor, predict_image

GRADIENT_FIX = {'released': 'topk_loss = torch.mean((torch.tensor(diff_k)).clone().detach())', 'here': 'topk_loss = torch.mean(diff_k)'}
LAMBDA_CYCLE = 10


def top_k(a, b):
    """The release's Top-k term (``topKloss``) with its gradient: the mean of the largest tenth of |a - b|."""
    diff = torch.abs(a-b).flatten()
    return torch.topk(diff, int(.1*diff.numel()), sorted=True)[0].mean()


def train(e):
    a = e.args; torch.set_num_threads(4); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    root = upstream_of(a.upstream, a.model)
    Generator = module_file('official_vcgan_generator', root/'generator_model.py').Generator
    Discriminator = module_file('official_vcgan_discriminator', root/'discriminator_model.py').Discriminator
    disc_a = Discriminator(in_channels=3).cuda(); disc_b = Discriminator(in_channels=3).cuda()
    gen_b = Generator(img_channels=3, num_residuals=9).cuda(); gen_a = Generator(img_channels=3, num_residuals=9).cuda()
    lr = 1e-5*a.learning_rate_scale
    opt_disc = torch.optim.Adam(list(disc_a.parameters())+list(disc_b.parameters()), lr=lr, betas=(.5, .999))
    opt_gen = torch.optim.Adam(list(gen_b.parameters())+list(gen_a.parameters()), lr=lr, betas=(.5, .999))
    d_scaler = torch.amp.GradScaler('cuda'); g_scaler = torch.amp.GradScaler('cuda')
    config = {'architecture': 'official ResNet generators (9 blocks), PatchGAN critics, 3 channels', 'variant': 'VCGAN, gradient fix (修梯度)',
              'gradient_fix': GRADIENT_FIX, 'loss': 'LSGAN MSE + 10 Top-k (10%) cycle each way; cycle L1 and identity off, as released',
              'lr': lr, 'betas': [.5, .999], 'batch': a.batch_size, 'precision': 'float16 autocast, GradScaler', 'framework': torch.__version__,
              'parameters': sum(p.numel() for p in gen_a.parameters()), 'paired_training': False,
              'adaptation': 'grey native 256 tiles repeated to 3 channels in [-1, 1]; independent domains from the common training pool; seed42'}
    write_json(e.out/'model_config.json', config); e.log(config)

    def predict(x): return predict_image(gen_a, None, x, a.model)

    def save(path):
        path.mkdir()
        torch.save({'model': gen_a.state_dict(), 'inverse': gen_b.state_dict(), 'disc_a': disc_a.state_dict(), 'disc_b': disc_b.state_dict(),
                    'generator_optimizer': opt_gen.state_dict(), 'discriminator_optimizer': opt_disc.state_dict(), 'step': step, 'config': config,
                    'batch_rng': e.rng.bit_generator.state, 'unpaired_rng': e.unpaired_rng.bit_generator.state,
                    'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state()}, path/'weights.pt')
        restored = torch.load(path/'weights.pt', map_location='cpu', weights_only=False)['model']
        require(all(torch.equal(v.detach().cpu(), restored[k]) for k, v in gen_a.state_dict().items()), 'checkpoint reload mismatch')

    for step in range(1, a.steps+1):
        x, y = e.batch(unpaired=True); b_real = image_tensor(x, a.model); a_real = image_tensor(y, a.model)
        for m in (disc_a, disc_b, gen_a, gen_b): m.train()
        with torch.autocast('cuda'):
            fake_a = gen_a(b_real); real_a, guess_a = disc_a(a_real), disc_a(fake_a.detach())
            fake_b = gen_b(a_real); real_b, guess_b = disc_b(b_real), disc_b(fake_b.detach())
            dl = (F.mse_loss(real_a, torch.ones_like(real_a))+F.mse_loss(guess_a, torch.zeros_like(guess_a))
                  + F.mse_loss(real_b, torch.ones_like(real_b))+F.mse_loss(guess_b, torch.zeros_like(guess_b)))/2
        require(torch.isfinite(dl).item(), 'nonfinite VCGAN critic loss')
        opt_disc.zero_grad(); d_scaler.scale(dl).backward(); d_scaler.step(opt_disc); d_scaler.update()
        with torch.autocast('cuda'):
            judged_a, judged_b = disc_a(fake_a), disc_b(fake_b)
            adversarial = F.mse_loss(judged_a, torch.ones_like(judged_a))+F.mse_loss(judged_b, torch.ones_like(judged_b))
            gl = adversarial+LAMBDA_CYCLE*top_k(a_real, gen_a(fake_b))+LAMBDA_CYCLE*top_k(b_real, gen_b(fake_a))
        require(torch.isfinite(gl).item(), 'nonfinite VCGAN generator loss')
        opt_gen.zero_grad(); g_scaler.scale(gl).backward(); g_scaler.step(opt_gen); g_scaler.update()
        if step == 1 or step % 100 == 0:
            e.log({'step': step, 'generator_loss': float(gl.detach()), 'discriminator_loss': float(dl.detach()),
                   'peak_gpu_mib': torch.cuda.max_memory_allocated()/1024**2})
        if step % a.eval_every == 0 or step == a.steps:
            e.evaluate(predict, step, save)
            if e.stop: break
    e.finish()
