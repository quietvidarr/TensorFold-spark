"""The EXL3 n-gram table's explicit reads (TF_EXL3_NGRAM_READ=pread) return the mapped rows byte for byte."""

from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.torch


def _table(tmp_path, monkeypatch, mode, readers, bits=6, rows=(700, 1300), head=88):
    import torch

    from tensorfold.families.qwen4_exp.cuda import exl3_pack

    monkeypatch.setenv(exl3_pack.NGRAM_READ, mode)
    monkeypatch.setenv(exl3_pack.NGRAM_READERS, str(readers))
    words = 1 + 160 * bits // 16
    data = np.random.default_rng(bits).integers(-2**15, 2**15, size=(sum(rows), words), dtype=np.int16)
    (tmp_path / "ngram.safetensors").write_bytes(bytes(head) + data.tobytes())   # rows after a header, as in a pack
    entries = {f"t.shard_{i}.trellis": ("ngram.safetensors", head + 2 * words * sum(rows[:i]),
                                        head + 2 * words * sum(rows[:i + 1]), "I16", [n, words])
               for i, n in enumerate(rows)}
    tensors = {"t.head_bias": torch.zeros(4), "t.head_offsets": torch.zeros(2, dtype=torch.int64),
               "t.head_vocab_sizes": torch.ones(2, dtype=torch.int64), "t.layer_multipliers": torch.ones(2)}
    pk = SimpleNamespace(dir=tmp_path, entry=entries.__getitem__, get=tensors.__getitem__)
    return exl3_pack.NgramTable(pk, "t.", len(rows), "cpu"), data


@pytest.mark.parametrize("readers", [1, 4, 32])
def test_pread_rows_match_the_mapped_rows(tmp_path, monkeypatch, readers):
    table, data = _table(tmp_path, monkeypatch, "pread", readers)
    rng = np.random.default_rng(readers)
    for ids in (rng.integers(0, len(data), size=(2048, 16)), np.arange(len(data)), np.array([699, 700, 700, 1999]),
                np.array([5]), np.arange(len(data))[::-1]):
        got = table.gather(ids)
        assert got.shape == (ids.size, data.shape[1]) and got.dtype == np.int16
        assert got.tobytes() == data[ids.reshape(-1)].tobytes()
    assert table.gather([]).shape == (0, data.shape[1])
    for bad in ([-1], [len(data)]):
        with pytest.raises(IndexError, match="outside"):
            table.gather(bad)


def test_pread_and_mmap_modes_agree(tmp_path, monkeypatch):
    ids = np.random.default_rng(0).integers(0, 2000, size=(512, 16))
    a, _ = _table(tmp_path, monkeypatch, "pread", 8)
    b, _ = _table(tmp_path, monkeypatch, "mmap", 8)
    assert a.gather(ids).tobytes() == b.gather(ids).tobytes()


def test_an_unknown_read_mode_is_refused(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="TF_EXL3_NGRAM_READ"):
        _table(tmp_path, monkeypatch, "fast", 8)


def test_stage_ple_uses_rows_read_ahead(tmp_path, monkeypatch):
    import torch

    from tensorfold.families.qwen4_exp.cuda import exl3_pack

    table, data = _table(tmp_path, monkeypatch, "pread", 8)
    ids = np.array([[3, 1999], [700, 4]])
    sc = SimpleNamespace(ple_host=torch.zeros(8, data.shape[1], dtype=torch.int16),
                         ple_dev=torch.zeros(8, data.shape[1], dtype=torch.int16))
    exl3_pack.stage_ple(table, sc, ids, at=2, rows=table.gather(ids))
    assert sc.ple_dev[2:6].numpy().tobytes() == data[ids.reshape(-1)].tobytes()
