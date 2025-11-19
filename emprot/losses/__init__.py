#!/usr/bin/env python3
"""
Slim losses API for classification-only training.
"""

from .masked_ce import masked_cross_entropy
from .histogram_ce import histogram_ce_loss
from .distributional import (
    per_residue_histogram_from_ids,
    aggregated_probability_kl_loss,
    kl_from_histograms,
    js_from_histograms,
    residue_centric_loss,
    straight_through_gumbel_softmax,
    st_gumbel_hist_kl_loss,
    js_divergence,
    dwell_rate_loss_from_logits,
    transition_row_js_loss_from_logits,
    coverage_hinge_loss,
)

__all__ = [
    'masked_cross_entropy',
    'histogram_ce_loss',
    'per_residue_histogram_from_ids',
    'aggregated_probability_kl_loss',
    'kl_from_histograms',
    'js_from_histograms',
    'residue_centric_loss',
    'straight_through_gumbel_softmax',
    'st_gumbel_hist_kl_loss',
    'js_divergence',
    'dwell_rate_loss_from_logits',
    'transition_row_js_loss_from_logits',
    'coverage_hinge_loss',
]
