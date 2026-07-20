import json

from tinker_finetune.config import get_settings
from tinker_finetune.metrics import (
    JsonlLogger,
    NoOpLogger,
    as_callback,
    build_logger,
)
from tinker_finetune.models.schemas import TrainMetrics


def test_jsonl_logger_writes_lines(tmp_path):
    logger = JsonlLogger(tmp_path)
    logger.log(TrainMetrics(step=1, loss=1.5))
    logger.log(TrainMetrics(step=2, loss=1.2))
    logger.finish()
    lines = (tmp_path / "metrics.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["loss"] == 1.5


def test_build_logger_defaults_to_jsonl_when_no_wandb(tmp_path):
    settings = get_settings()  # dry-run, no wandb project
    logger = build_logger(settings, run_dir=tmp_path)
    assert isinstance(logger, JsonlLogger)
    logger.finish()


def test_build_logger_noop_without_run_dir():
    settings = get_settings()
    assert isinstance(build_logger(settings), NoOpLogger)


def test_as_callback_fans_out(tmp_path):
    logger = JsonlLogger(tmp_path)
    seen = []
    cb = as_callback(logger, seen.append)
    cb(TrainMetrics(step=1, loss=0.9))
    logger.finish()
    assert seen and seen[0].step == 1
    assert (tmp_path / "metrics.jsonl").read_text().strip()
