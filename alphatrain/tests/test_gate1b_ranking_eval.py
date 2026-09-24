import pytest
import torch

from alphatrain.scripts.gate1b_ranking_eval import head_output_to_scalar


def test_survival_logits_are_sigmoided_before_weighting():
    logits = torch.zeros(1, 4)
    value = head_output_to_scalar(logits, 'value_head', 'survival')
    assert value.item() == pytest.approx(0.5 * (1.0 + 0.8 + 0.5 + 0.25))


def test_density_outputs_use_density_weights_without_sigmoid():
    outputs = torch.tensor([[1.0, 2.0, 3.0]])
    value = head_output_to_scalar(outputs, 'value_head', 'density')
    assert value.item() == pytest.approx(1.0 * 0.5 + 2.0 * 0.3 + 3.0 * 0.2)


def test_scalar_ranking_head_is_raw():
    output = torch.tensor([[2.75]])
    value = head_output_to_scalar(output, 'spatial', 'pairwise_ranking')
    assert value.item() == pytest.approx(2.75)
