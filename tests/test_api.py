import time

import pytest
from fastapi.testclient import TestClient

from tinker_finetune.api.app import create_app


@pytest.fixture
def client():
    return TestClient(create_app())


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["dry_run"] is True


def test_list_models(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    names = {m["name"] for m in r.json()}
    assert "Qwen/Qwen3-8B" in names


def test_model_detail_404_for_closed_model(client):
    r = client.get("/v1/models/acme/closed")
    assert r.status_code == 404


def test_sample_endpoint(client):
    r = client.post("/v1/inference/sample", json={
        "base_model": "Qwen/Qwen3-8B",
        "prompt": "hello",
        "max_new_tokens": 16,
    })
    assert r.status_code == 200
    assert "text" in r.json()


def test_dataset_inspect(client, sample_dataset):
    r = client.get("/v1/datasets/inspect", params={"path": sample_dataset})
    assert r.status_code == 200
    assert r.json()["num_examples"] == 2


def test_sft_job_lifecycle(client, sample_dataset):
    r = client.post("/v1/jobs/sft", json={
        "base_model": "Qwen/Qwen3-8B",
        "train_path": sample_dataset,
        "epochs": 1,
        "batch_size": 1,
        "max_seq_len": 64,
    })
    assert r.status_code == 201
    job_id = r.json()["job"]["id"]

    # Poll until terminal (fake backend is fast).
    for _ in range(100):
        status = client.get(f"/v1/jobs/{job_id}").json()["status"]
        if status in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(0.05)
    assert status == "succeeded"

    metrics = client.get(f"/v1/jobs/{job_id}/metrics").json()
    assert metrics["current_step"] > 0
    assert metrics["metrics"]


@pytest.fixture
def preference_dataset(tmp_path):
    p = tmp_path / "prefs.jsonl"
    p.write_text(
        '{"prompt": "q1", "chosen": "a good detailed answer", "rejected": "no"}\n'
        '{"prompt": "q2", "chosen": "another helpful reply", "rejected": "x"}\n'
    )
    return str(p)


def test_dpo_job_lifecycle(client, preference_dataset):
    r = client.post("/v1/jobs/dpo", json={
        "base_model": "thinkingmachines/Inkling",
        "train_path": preference_dataset,
        "epochs": 1,
        "batch_size": 1,
        "max_seq_len": 128,
    })
    assert r.status_code == 201
    job_id = r.json()["job"]["id"]

    for _ in range(100):
        status = client.get(f"/v1/jobs/{job_id}").json()["status"]
        if status in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(0.05)
    assert status == "succeeded"

    metrics = client.get(f"/v1/jobs/{job_id}/metrics").json()
    assert metrics["current_step"] > 0
    assert any(m.get("reward_accuracy") is not None for m in metrics["metrics"])


def test_rl_job_with_reward_model(client):
    r = client.post("/v1/jobs/rl", json={
        "base_model": "Qwen/Qwen3-8B",
        "prompts": ["explain gradients", "write a greeting"],
        "reward": "rm:Qwen/Qwen3-8B",
        "iterations": 2,
        "group_size": 2,
        "prompts_per_batch": 2,
        "max_new_tokens": 8,
    })
    assert r.status_code == 201
    job_id = r.json()["job"]["id"]
    for _ in range(100):
        status = client.get(f"/v1/jobs/{job_id}").json()["status"]
        if status in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(0.05)
    assert status == "succeeded"


def test_inkling_in_model_list(client):
    r = client.get("/v1/models", params={"family": "inkling"})
    assert r.status_code == 200
    names = {m["name"] for m in r.json()}
    assert "thinkingmachines/Inkling" in names


def test_sft_job_rejects_closed_model(client, sample_dataset):
    r = client.post("/v1/jobs/sft", json={
        "base_model": "acme/closed",
        "train_path": sample_dataset,
    })
    assert r.status_code == 400


def test_dataset_validate_endpoint_reports_findings(client, tmp_path):
    dirty = tmp_path / "dirty.jsonl"
    dirty.write_text(
        '{"prompt": "contact", "completion": "reach me at ops@example.com"}\n'
        '{"prompt": "contact", "completion": "reach me at ops@example.com"}\n'
    )
    r = client.get("/v1/datasets/validate", params={"path": str(dirty)})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    codes = {f["code"] for f in body["findings"]}
    assert "pii_detected" in codes and "exact_duplicates" in codes
    assert body["stats"]["duplicate_rate"] > 0


def test_dataset_validate_detects_eval_contamination(client, tmp_path):
    train = tmp_path / "train.jsonl"
    train.write_text('{"prompt": "capital of france", "completion": "paris"}\n')
    evals = tmp_path / "eval.jsonl"
    evals.write_text('{"prompt": "capital of france", "completion": "paris"}\n')
    r = client.get(
        "/v1/datasets/validate", params={"path": str(train), "eval_path": str(evals)}
    )
    assert r.status_code == 200
    assert "eval_contamination" in {f["code"] for f in r.json()["findings"]}


def test_dataset_validate_missing_file_is_a_400(client, tmp_path):
    r = client.get("/v1/datasets/validate", params={"path": str(tmp_path / "nope.jsonl")})
    assert r.status_code == 400


def test_dataset_validate_clean_dataset_passes(client, tmp_path):
    clean = tmp_path / "clean.jsonl"
    clean.write_text(
        "".join(
            f'{{"prompt": "question {i} about widgets", "completion": "answer {i}"}}\n'
            for i in range(12)
        )
    )
    body = client.get("/v1/datasets/validate", params={"path": str(clean)}).json()
    assert body["ok"] is True
    assert body["num_examples"] == 12
