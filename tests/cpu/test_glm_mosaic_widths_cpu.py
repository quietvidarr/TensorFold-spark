"""CPU contracts for the packed GLM mosaic widths: 2-bit and 3-bit routed experts admitted, others refused."""
import torch
import pytest

from tensorfold.cuda.exl3 import experts as generic_exl3
from tensorfold.families.glm5_next.cuda import weights


def trellis(k2: int) -> torch.Tensor:
    # [K/16, N/16, 16*K] int16 with K = k2/2 bits; widths only, contents irrelevant
    return torch.zeros((4, 4, 8 * k2), dtype=torch.int16)


def test_admitted_widths_are_two_and_three_bits():
    assert weights.PACKED_EXPERT_K2 == (4, 6)
    assert generic_exl3.k2_of(trellis(4)) == 4
    assert generic_exl3.k2_of(trellis(6)) == 6


@pytest.mark.parametrize("k2", [2, 3, 5, 7, 8, 10])
def test_other_widths_stay_outside_the_packed_profile(k2):
    assert generic_exl3.k2_of(trellis(k2)) not in weights.PACKED_EXPERT_K2


def test_gate_lives_in_loader_source():
    import inspect
    src = inspect.getsource(weights)
    assert "k2 not in PACKED_EXPERT_K2" in src
    assert "expected 2-bit compressed expert" not in src
