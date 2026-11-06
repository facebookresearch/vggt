import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from trainer import chunk_batch_for_accum_steps


@pytest.mark.parametrize("size,requested", [(1, 4), (2, 4), (4, 2)])
def test_accumulation_does_not_create_empty_microbatches(size, requested):
    images = torch.arange(size, dtype=torch.float32).view(size, 1)
    names = [f"seq-{i}" for i in range(size)]
    chunks = chunk_batch_for_accum_steps({"images": images, "seq_name": names}, requested)
    assert len(chunks) == min(size, requested)
    assert all(len(chunk["images"]) > 0 for chunk in chunks)
    torch.testing.assert_close(torch.cat([chunk["images"] for chunk in chunks]), images)
    assert [name for chunk in chunks for name in chunk["seq_name"]] == names
