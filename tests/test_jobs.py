import time

from tinker_finetune.jobs.manager import JobManager
from tinker_finetune.models.schemas import JobStatus, JobType, RLConfig, SFTConfig


def _wait(manager, job_id, timeout=8.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        rec = manager.get(job_id)
        if rec and rec.status in (JobStatus.succeeded, JobStatus.failed, JobStatus.cancelled):
            return rec
        time.sleep(0.03)
    return manager.get(job_id)


def test_sft_job_succeeds(sample_dataset):
    manager = JobManager()
    cfg = SFTConfig(base_model="Qwen/Qwen3-8B", epochs=1, batch_size=1, max_seq_len=64)
    rec = manager.submit(JobType.sft, cfg, train_path=sample_dataset)
    rec = _wait(manager, rec.id)
    assert rec.status == JobStatus.succeeded
    assert rec.current_step > 0
    assert rec.finished_at is not None


def test_rl_job_succeeds():
    manager = JobManager()
    cfg = RLConfig(base_model="Qwen/Qwen3-8B", iterations=2, group_size=2,
                   prompts_per_batch=2, max_new_tokens=8)
    rec = manager.submit(JobType.rl, cfg, prompts=["a", "b"], reward="nonempty")
    rec = _wait(manager, rec.id)
    assert rec.status == JobStatus.succeeded


def test_missing_train_path_fails_job():
    manager = JobManager()
    cfg = SFTConfig(base_model="Qwen/Qwen3-8B", epochs=1)
    rec = manager.submit(JobType.sft, cfg, train_path=None)
    rec = _wait(manager, rec.id)
    assert rec.status == JobStatus.failed
    assert rec.error


def test_job_state_persisted_and_reloaded(sample_dataset):
    manager = JobManager()
    cfg = SFTConfig(base_model="Qwen/Qwen3-8B", epochs=1, batch_size=1, max_seq_len=64)
    rec = manager.submit(JobType.sft, cfg, train_path=sample_dataset)
    _wait(manager, rec.id)
    # A fresh manager over the same artifacts dir reloads the record.
    reloaded = JobManager()
    assert reloaded.get(rec.id) is not None
