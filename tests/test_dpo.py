import math

from tinker_finetune.data.preferences import (
    build_preference_datums,
    load_preference_dataset,
)
from tinker_finetune.data.tokenization import ByteTokenizer
from tinker_finetune.models.schemas import DPOConfig, Message, PreferenceExample, Role
from tinker_finetune.tinker_client.fake import FakeTinkerBackend
from tinker_finetune.training.dpo_trainer import DPOTrainer, _log_sigmoid


def _prefs(n=6):
    return [
        PreferenceExample(
            prompt=[Message(role=Role.user, content=f"question {i}")],
            chosen=f"a thorough, helpful answer number {i} with detail",
            rejected="no",
        )
        for i in range(n)
    ]


def test_log_sigmoid_matches_reference():
    for x in (-4.0, -0.5, 0.0, 0.5, 4.0):
        assert math.isclose(_log_sigmoid(x), math.log(1 / (1 + math.exp(-x))), rel_tol=1e-9)


def test_preference_datums_mask_prompt():
    tok = ByteTokenizer()
    pref = PreferenceExample(
        prompt=[Message(role=Role.user, content="hi")],
        chosen="a good detailed reply",
        rejected="no",
    )
    chosen, rejected = build_preference_datums(pref, tok, max_seq_len=1000)
    assert chosen.num_supervised_tokens > 0
    assert rejected.num_supervised_tokens > 0
    # Chosen response is longer, so it supervises more tokens than rejected.
    assert chosen.num_supervised_tokens > rejected.num_supervised_tokens


def test_dpo_loss_prefers_chosen_when_policy_favours_it():
    cfg = DPOConfig(base_model="Qwen/Qwen3-8B", beta=0.1)
    trainer = DPOTrainer(
        FakeTinkerBackend("Qwen/Qwen3-8B", cfg.lora), None, ByteTokenizer(), cfg
    )
    # policy strongly prefers chosen over the reference-adjusted rejected
    loss, margin, preferred = trainer._dpo_loss(pol_c=-2.0, pol_r=-5.0, ref_c=-3.0, ref_r=-3.0)
    assert preferred is True
    assert margin > 0
    assert loss < math.log(2)  # better than a coin flip


def test_dpo_trainer_runs_and_reports_accuracy():
    cfg = DPOConfig(base_model="Qwen/Qwen3-8B", epochs=2, batch_size=2, max_seq_len=128)
    backend = FakeTinkerBackend(cfg.base_model, cfg.lora)
    reference = FakeTinkerBackend(cfg.base_model, cfg.lora)
    trainer = DPOTrainer(backend, reference, ByteTokenizer(), cfg)
    history = trainer.train(_prefs())
    assert len(history) > 0
    assert all(m.reward_accuracy is not None for m in history)
    assert all(m.reward_margin is not None for m in history)


def test_dpo_reference_free_mode():
    cfg = DPOConfig(base_model="Qwen/Qwen3-8B", epochs=1, batch_size=2, reference_free=True)
    trainer = DPOTrainer(
        FakeTinkerBackend(cfg.base_model, cfg.lora), None, ByteTokenizer(), cfg
    )
    history = trainer.train(_prefs(4))
    assert len(history) > 0


def test_load_preference_dataset(tmp_path):
    p = tmp_path / "prefs.jsonl"
    p.write_text(
        '{"prompt": "q", "chosen": "good", "rejected": "bad"}\n'
        '{"prompt": [{"role": "user", "content": "q2"}], "chosen": "c", "rejected": "r"}\n'
    )
    prefs = load_preference_dataset(p)
    assert len(prefs) == 2
    assert prefs[0].chosen == "good"
