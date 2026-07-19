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


def test_sft_job_rejects_closed_model(client, sample_dataset):
    r = client.post("/v1/jobs/sft", json={
        "base_model": "acme/closed",
        "train_path": sample_dataset,
    })
    assert r.status_code == 400
