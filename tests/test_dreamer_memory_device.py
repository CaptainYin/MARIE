import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.memory.DreamerMemory import DreamerMemory
from environments import Env


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA device")
def test_attention_mask_is_created_on_done_tensor_device():
    memory = DreamerMemory(
        capacity=10,
        sequence_length=3,
        action_size=2,
        obs_size=4,
        n_agents=2,
        device="cuda",
        env_type=Env.PETTINGZOO,
    )
    dones = torch.zeros(2, 3, 2, 1, device="cuda")
    dones[1, 1, :, 0] = 1.0

    mask = memory.generate_attn_mask(dones, tokens_per_block=2)

    assert mask.device == dones.device
    assert mask.shape == (4, 6, 6)
