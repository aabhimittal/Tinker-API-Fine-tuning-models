import pytest

from tinker_finetune.data.tokenization import ByteTokenizer
from tinker_finetune.models.schemas import (
    ChatExample,
    LoRAConfig,
    Message,
    OptimConfig,
    RLConfig,
    Role,
    SFTConfig,
)
from tinker_finetune.rewards import get_reward_fn
from tinker_finetune.tinker_client.fake import FakeTinkerBackend
from tinker_finetune.training.optim import lr_at_step
from tinker_finetune.training.rl_trainer import RLTrainer, _group_advantages
from tinker_finetune.training.sft_trainer import SFTTrainer


def _examples(n=8):
    return [
        ChatExample(messages=[
            Message(role=Role.user, content=f"question {i}"),
            Message(role=Role.assistant, content=f"answer number {i} here"),
        ])
        for i in range(n)
    ]


def test_lr_warmup_then_decay():
    cfg = OptimConfig(learning_rate=1e-3, warmup_ratio=0.1, lr_schedule="cosine", min_lr_ratio=0.1)
    lrs = [lr_at_step(cfg, s, 100) for s in range(100)]
    assert lrs[0] < lrs[10]           # warming up
    assert lrs[10] == max(lrs)        # peak near end of warmup
    assert lrs[-1] < lrs[10]          # decayed
    assert lrs[-1] >= 1e-3 * 0.1 - 1e-9


def test_constant_schedule_is_flat_after_warmup():
    cfg = OptimConfig(learning_rate=5e-4, warmup_ratio=0.0, lr_schedule="constant")
    lrs = [lr_at_step(cfg, s, 50) for s in range(50)]
    assert max(lrs) == pytest.approx(5e-4)
    assert min(lrs) == pytest.approx(5e-4)


def test_sft_trainer_produces_decreasing_loss_trend():
    cfg = SFTConfig(base_model="Qwen/Qwen3-8B", epochs=3, batch_size=2, max_seq_len=64)
    backend = FakeTinkerBackend(cfg.base_model, cfg.lora)
    trainer = SFTTrainer(backend, ByteTokenizer(), cfg)
    history = trainer.train(_examples())
    assert len(history) > 0
    # Fake backend trends down; compare first vs last third averages.
    third = max(1, len(history) // 3)
    early = sum(m.loss for m in history[:third]) / third
    late = sum(m.loss for m in history[-third:]) / third
    assert late < early


def test_sft_trainer_stop_flag_halts_early():
    cfg = SFTConfig(base_model="Qwen/Qwen3-8B", epochs=5, batch_size=1, max_seq_len=64)
    backend = FakeTinkerBackend(cfg.base_model, cfg.lora)
    calls = {"n": 0}

    def should_stop():
        calls["n"] += 1
        return calls["n"] > 3

    trainer = SFTTrainer(backend, ByteTokenizer(), cfg, should_stop=should_stop)
    history = trainer.train(_examples(4))
    assert len(history) <= 3


def test_group_advantages_grpo_zero_mean():
    adv = _group_advantages([1.0, 2.0, 3.0], "grpo")
    assert abs(sum(adv)) < 1e-6            # standardized -> ~zero mean
    adv_r = _group_advantages([1.0, 2.0, 3.0], "reinforce")
    assert adv_r == [-1.0, 0.0, 1.0]


def test_rl_trainer_runs_and_reports_reward():
    cfg = RLConfig(base_model="Qwen/Qwen3-8B", iterations=3, group_size=4,
                   prompts_per_batch=2, max_new_tokens=8, lora=LoRAConfig(rank=8))
    backend = FakeTinkerBackend(cfg.base_model, cfg.lora)
    trainer = RLTrainer(backend, ByteTokenizer(), cfg, get_reward_fn("nonempty"))
    history = trainer.train(["prompt a", "prompt b", "prompt c"])
    assert len(history) == 3
    assert all(m.reward_mean is not None for m in history)
