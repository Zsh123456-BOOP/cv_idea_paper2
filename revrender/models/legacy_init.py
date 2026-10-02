"""Modules built only to consume random numbers.

Every RevRender model trained so far was initialised from a stream that had first built a larger, three-part model: its
``see`` was kept, and its ``find`` (a seed-selected query decoder) and ``draw`` (a competition renderer) were discarded
after the detector and renderer in models/detector.py and models/renderer.py replaced them.  Those two discarded modules
drew from the global random stream, so every weight built after them (the mask-quality head) and every CUDA draw
that follows depends on them having been built.  They are built here, exactly as before, and thrown away.

Dropping them gives a different (equally valid) initialisation and a model that no longer replays the paper-1 training
trace.  Do not tidy them away without retraining and re-checking the smoke trace (tests/ and README: training equivalence).
"""
import torch
from torch import nn

from .backbone import SeeDetails

DIMENSION, CHANNELS, QUERIES, LAYERS = 160, 32, 32, 4


class _UnusedFinder(nn.Module):
    def __init__(self, dimension=DIMENSION, channels=CHANNELS, queries=QUERIES, layers=LAYERS):
        super().__init__()
        self.query_count = queries
        self.position = nn.Sequential(nn.Linear(2, dimension), nn.ReLU(), nn.Linear(dimension, dimension))
        self.queries = nn.Parameter(torch.randn(queries, dimension)*.02)
        self.layers = nn.ModuleList([nn.TransformerDecoderLayer(
            dimension, 4, dimension*4, dropout=0., batch_first=True, norm_first=True) for _ in range(layers)])
        self.mask_embed = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, channels))
        self.objectness = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, 1))
        self.seed_fine_projection = nn.Linear(channels, dimension, bias=False)
        nn.init.constant_(self.objectness[-1].bias, -2.)


class _UnusedDraw(nn.Module):
    def __init__(self, dimension=DIMENSION, channels=CHANNELS):
        super().__init__()
        self.query_code = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, 16))
        self.local_token = nn.Sequential(nn.Linear(channels+16+4+1, 64), nn.GELU(), nn.Linear(64, 32))
        self.local_delta = nn.Linear(32, 1)
        self.local_gate = nn.Parameter(torch.tensor(-2.))
        self.background_token = nn.Linear(channels, 32)
        self.comp_norm = nn.LayerNorm(32)
        self.comp_attention = nn.MultiheadAttention(32, 4, dropout=0., batch_first=True)
        self.comp_delta = nn.Linear(32, 1)
        self.comp_gate = nn.Parameter(torch.tensor(-2.))
        self.owner_mix = nn.Parameter(torch.tensor(-2.))
        nn.init.normal_(self.local_delta.weight, std=.01)
        nn.init.zeros_(self.local_delta.bias)
        nn.init.normal_(self.comp_delta.weight, std=.01)
        nn.init.zeros_(self.comp_delta.bias)


def build_see_consuming_legacy_draws(segmenter):
    """The see stage on the segmenter's encoder and decoder, built after (and in the stream of) the discarded modules'
    constructors in the original order: see, then the discarded finder, then the discarded renderer."""
    see = SeeDetails(segmenter.encoder, segmenter.decoder, int(segmenter.segmentation_head[0].in_channels))
    _UnusedFinder()
    _UnusedDraw()
    return see
