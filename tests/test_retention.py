"""Edge cases for checkpoint retention: ties, missing metrics, corrupt manifests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tinker_finetune.training.checkpoint import CheckpointManager
from tinker_finetune.training.retention import (
    CheckpointEntry,
    CorruptManifestError,
    RetentionPolicy,
    apply_retention,
    plan_retention,
)


def entries(*specs) -> list[dict]:
    return [{"step": step, "path": f"/tmp/step-{step}.state", "metrics": metrics} for step, metrics in specs]


def kept(plan) -> list[int]:
    return sorted(e.step for e in plan.keep)


def test_keep_last_and_best_are_a_union():
    plan = plan_retention(
        entries((100, {"eval_loss": 0.5}), (200, {"eval_loss": 0.1}), (300, {"eval_loss": 0.9}), (400, {"eval_loss": 0.8})),
        RetentionPolicy(keep_last=2, keep_best=1),
    )
    assert kept(plan) == [200, 300, 400]
    assert "best" in plan.reasons[200]
    assert "final" in plan.reasons[400]


def test_ties_break_toward_the_later_step():
    for mode in ("min", "max"):
        plan = plan_retention(
            entries((10, {"acc": 0.7}), (20, {"acc": 0.7}), (30, {"acc": 0.1 if mode == "max" else 0.9})),
            RetentionPolicy(keep_last=0, keep_best=1, metric="acc", mode=mode, protect_final=False),
        )
        assert kept(plan) == [20], mode


def test_entries_without_the_metric_never_win_best():
    plan = plan_retention(
        entries((10, {}), (20, {"eval_loss": 2.0}), (30, {})),
        RetentionPolicy(keep_last=0, keep_best=1, protect_final=False),
    )
    assert kept(plan) == [20]


def test_nan_and_infinite_metrics_are_ignored():
    raw = [
        {"step": 10, "path": "a", "metrics": {"eval_loss": float("nan")}},
        {"step": 20, "path": "b", "metrics": {"eval_loss": float("-inf")}},
        {"step": 30, "path": "c", "metrics": {"eval_loss": 1.5}},
    ]
    plan = plan_retention(raw, RetentionPolicy(keep_last=0, keep_best=1, protect_final=False))
    assert kept(plan) == [30]


def test_boolean_metrics_are_not_treated_as_numbers():
    entry = CheckpointEntry.from_dict({"step": 1, "path": "p", "metrics": {"converged": True, "loss": 0.2}})
    assert entry.metrics == {"loss": 0.2}


def test_keep_every_builds_a_bisect_ladder():
    plan = plan_retention(
        entries(*[(step, {}) for step in range(100, 1100, 100)]),
        RetentionPolicy(keep_last=1, keep_best=0, keep_every=500),
    )
    assert kept(plan) == [500, 1000]


def test_min_age_protects_recent_checkpoints():
    plan = plan_retention(
        entries(*[(step, {}) for step in (100, 200, 300, 400)]),
        RetentionPolicy(keep_last=0, keep_best=0, keep_every=0, min_age_steps=250, protect_final=True),
    )
    assert kept(plan) == [200, 300, 400]


def test_duplicate_steps_collapse_to_the_last_write():
    plan = plan_retention(
        [
            {"step": 10, "path": "old.state", "metrics": {}},
            {"step": 10, "path": "new.state", "metrics": {}},
        ],
        RetentionPolicy(keep_last=1),
    )
    assert [e.path for e in plan.keep] == ["new.state"]


def test_empty_manifest_is_a_no_op():
    plan = plan_retention([], RetentionPolicy())
    assert plan.keep == [] and plan.delete == []


def test_single_checkpoint_is_never_deleted():
    plan = plan_retention(entries((7, {})), RetentionPolicy(keep_last=1))
    assert kept(plan) == [7]


@pytest.mark.parametrize(
    "kwargs",
    [{"keep_last": -1}, {"keep_best": -1}, {"keep_every": -1}, {"min_age_steps": -1}, {"mode": "lowest"}],
)
def test_invalid_policies_are_rejected(kwargs):
    with pytest.raises(ValueError):
        RetentionPolicy(**kwargs)


def write_manifest(tmp_path: Path, steps) -> Path:
    manifest = tmp_path / "checkpoints.json"
    payload = []
    for step, metrics in steps:
        ckpt = tmp_path / f"step-{step:06d}.state"
        ckpt.write_text("weights")
        payload.append({"step": step, "path": str(ckpt), "metrics": metrics})
    manifest.write_text(json.dumps(payload))
    return manifest


def test_apply_deletes_losers_and_rewrites_the_manifest(tmp_path):
    manifest = write_manifest(tmp_path, [(100, {"eval_loss": 0.4}), (200, {"eval_loss": 0.9}), (300, {"eval_loss": 0.8})])
    plan = apply_retention(manifest, RetentionPolicy(keep_last=1, keep_best=1))
    assert kept(plan) == [100, 300]
    assert not (tmp_path / "step-000200.state").exists()
    assert [e["step"] for e in json.loads(manifest.read_text())] == [100, 300]


def test_dry_run_changes_nothing(tmp_path):
    manifest = write_manifest(tmp_path, [(100, {}), (200, {}), (300, {})])
    before = manifest.read_text()
    plan = apply_retention(manifest, RetentionPolicy(keep_last=1, keep_best=0), dry_run=True)
    assert [e.step for e in plan.delete] == [100, 200]
    assert manifest.read_text() == before
    assert (tmp_path / "step-000100.state").exists()


def test_vanished_checkpoints_are_pruned_not_fatal(tmp_path):
    manifest = write_manifest(tmp_path, [(100, {}), (200, {}), (300, {})])
    (tmp_path / "step-000300.state").unlink()  # someone cleaned the disk by hand
    plan = apply_retention(manifest, RetentionPolicy(keep_last=2))
    assert [e.step for e in plan.missing] == [300]
    assert [e["step"] for e in json.loads(manifest.read_text())] == [100, 200]


def test_directory_checkpoints_are_removed_recursively(tmp_path):
    manifest = tmp_path / "checkpoints.json"
    payload = []
    for step in (100, 200):
        ckpt = tmp_path / f"step-{step}"
        (ckpt / "shard").mkdir(parents=True)
        (ckpt / "shard" / "weights.bin").write_text("w")
        payload.append({"step": step, "path": str(ckpt), "metrics": {}})
    manifest.write_text(json.dumps(payload))
    apply_retention(manifest, RetentionPolicy(keep_last=1, keep_best=0))
    assert not (tmp_path / "step-100").exists()
    assert (tmp_path / "step-200" / "shard" / "weights.bin").exists()


def test_corrupt_manifest_is_reported_clearly(tmp_path):
    manifest = tmp_path / "checkpoints.json"
    manifest.write_text("{not json")
    with pytest.raises(CorruptManifestError):
        apply_retention(manifest, RetentionPolicy())
    manifest.write_text('{"step": 1}')
    with pytest.raises(CorruptManifestError):
        apply_retention(manifest, RetentionPolicy())
    manifest.write_text('[{"path": "x"}]')
    with pytest.raises(CorruptManifestError):
        apply_retention(manifest, RetentionPolicy())


def test_missing_manifest_is_a_no_op(tmp_path):
    plan = apply_retention(tmp_path / "nope.json", RetentionPolicy())
    assert plan.keep == [] and plan.delete == []


def test_policy_that_would_delete_everything_is_refused(tmp_path):
    manifest = write_manifest(tmp_path, [(100, {})])
    with pytest.raises(ValueError):
        apply_retention(manifest, RetentionPolicy(keep_last=0, keep_best=0, keep_every=0, protect_final=False))


def test_manifest_rewrite_leaves_no_temp_files(tmp_path):
    manifest = write_manifest(tmp_path, [(100, {}), (200, {})])
    apply_retention(manifest, RetentionPolicy(keep_last=1, keep_best=0))
    assert [p.name for p in tmp_path.glob("*.tmp")] == []


class _Backend:
    """Minimal stand-in for the Tinker backend: writes a file where told."""

    def __init__(self) -> None:
        self.loaded: str | None = None

    def save_state(self, path: str) -> str:
        Path(path).write_text("weights")
        return path

    def load_state(self, path: str) -> None:
        self.loaded = path


def test_checkpoint_manager_enforces_retention_as_it_saves(tmp_path):
    manager = CheckpointManager(tmp_path, "job-1", retention=RetentionPolicy(keep_last=2, keep_best=1))
    backend = _Backend()
    for step in range(1, 7):
        manager.save(backend, step, metrics={"eval_loss": 1.0 / step if step != 3 else 0.01})
    index = json.loads((tmp_path / "job-1" / "checkpoints.json").read_text())
    steps = [e["step"] for e in index]
    assert steps == [3, 5, 6]  # best + last two
    assert manager.latest()["step"] == 6
    assert manager.resume(backend) == 6


def test_checkpoint_manager_without_a_policy_keeps_everything(tmp_path):
    manager = CheckpointManager(tmp_path, "job-2")
    backend = _Backend()
    for step in range(1, 5):
        manager.save(backend, step)
    assert manager.enforce_retention() is None
    assert len(json.loads((tmp_path / "job-2" / "checkpoints.json").read_text())) == 4
