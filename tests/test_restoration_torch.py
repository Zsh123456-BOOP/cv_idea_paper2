"""Regression checks for the published two-stage restoration output and loss."""
import pytest

torch=pytest.importorskip('torch')
pytest.importorskip('torchmetrics')

from revrender.baselines.restoration_torch import final_image, stain_loss


def test_stain_inference_returns_final_stage_without_stacking_intermediate():
    final=torch.ones(2,3,16,16)
    intermediate=torch.zeros_like(final)
    result=final_image([final,intermediate],'StainRestorer')
    assert result.shape==(2,3,16,16) and torch.equal(result,final)
    assert final_image(final,'Restormer') is final


def test_stain_loss_supervises_both_stages_and_accepts_perfect_reconstruction():
    target=torch.linspace(0,1,3*16*16).reshape(1,3,16,16)
    assert stain_loss([target,target],target).item()==pytest.approx(0,abs=1e-6)
    outputs=[(target*.8+.05).requires_grad_(),(target*.7+.1).requires_grad_()]
    loss=stain_loss(outputs,target)
    assert torch.isfinite(loss) and loss.item()>0
    loss.backward()
    assert all(value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum()>0 for value in outputs)
