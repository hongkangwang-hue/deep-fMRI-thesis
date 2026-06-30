"""
M2 derivatives loader + BOLD-only QC 单元测试。

用临时小 .hf5 构造 NaN / 零方差 / 正常体素，验证统一 mask 的剔除逻辑与持久化，
不触碰真实 95556 维大文件。
"""

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.fmri.derivatives import (
    compute_bold_only_mask, load_response, save_voxel_mask)


def _write_story(data_dir: Path, subject: str, story: str, arr: np.ndarray):
    # data_dir 语义同 config：直指 preprocessed_data，下接 subject/story.hf5
    p = data_dir / subject
    p.mkdir(parents=True, exist_ok=True)
    with h5py.File(p / f"{story}.hf5", "w") as f:
        f.create_dataset("data", data=arr)


def test_bold_only_mask_excludes_nan_and_zero_var(tmp_path):
    rng = np.random.default_rng(0)
    n_tr, n_vox = 20, 6
    # 基准：全有效
    a = rng.standard_normal((n_tr, n_vox))
    b = rng.standard_normal((n_tr, n_vox))
    # story a: 体素 2 全常数（零方差）
    a[:, 2] = 3.14
    # story b: 体素 4 含 NaN（仅在某故事出现也要被全局剔除）
    b[5, 4] = np.nan
    _write_story(tmp_path, "UTS03", "sa", a)
    _write_story(tmp_path, "UTS03", "sb", b)

    res = compute_bold_only_mask(tmp_path, "UTS03", ["sa", "sb"])
    assert res["n_voxels"] == n_vox
    # 体素 2（零方差）与 4（NaN）被剔除
    assert set(np.flatnonzero(~res["valid_mask"])) == {2, 4}
    assert res["n_kept"] == n_vox - 2
    assert res["n_excluded_nan"] == 1
    assert res["n_excluded_zero_var"] == 1
    assert res["keep_index"].tolist() == [0, 1, 3, 5]


def test_voxel_count_mismatch_raises(tmp_path):
    _write_story(tmp_path, "UTS03", "sa", np.ones((10, 5)))
    _write_story(tmp_path, "UTS03", "sb", np.ones((10, 6)))
    with pytest.raises(ValueError):
        compute_bold_only_mask(tmp_path, "UTS03", ["sa", "sb"])


def test_load_response_applies_columns(tmp_path):
    arr = np.arange(20 * 6, dtype=float).reshape(20, 6)
    _write_story(tmp_path, "UTS03", "sa", arr)
    cols = np.array([0, 2, 5])
    out = load_response(tmp_path, "UTS03", "sa", columns=cols)
    assert out.shape == (20, 3)
    assert np.allclose(out, arr[:, cols])


def test_save_voxel_mask_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    a = rng.standard_normal((15, 8)); a[:, 3] = 1.0  # 零方差
    _write_story(tmp_path, "UTS03", "sa", a)
    res = compute_bold_only_mask(tmp_path, "UTS03", ["sa"])
    paths = save_voxel_mask(tmp_path / "frozen", res,
                            provenance={"git": "test"})
    idx = np.load(paths["index_path"])
    meta = json.loads(Path(paths["meta_path"]).read_text())
    assert idx.tolist() == res["keep_index"].tolist()
    assert meta["n_kept"] == res["n_kept"]
    assert meta["provenance"]["git"] == "test"
    assert meta["qc_rule"]["exclude_zero_variance"] is True