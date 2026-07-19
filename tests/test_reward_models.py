from tinker_finetune.data.tokenization import ByteTokenizer
from tinker_finetune.models.schemas import LoRAConfig
from tinker_finetune.reward_models import (
    FakeRewardModel,
    TinkerRewardModel,
    build_reward_model,
    resolve_reward,
)
from tinker_finetune.tinker_client.fake import FakeTinkerBackend


def test_fake_reward_model_scores_bounded_and_prefers_substance():
    rm = FakeRewardModel()
    good = rm.score("q", "A clear, reasonably detailed and helpful response.")
    bad = rm.score("q", "")
    assert 0.0 <= good <= 1.0
    assert bad == 0.0
    assert good > bad


def test_tinker_reward_model_returns_probability():
    backend = FakeTinkerBackend("Qwen/Qwen3-8B", LoRAConfig(rank=8))
    rm = TinkerRewardModel(backend, ByteTokenizer())
    score = rm.score("what is 2+2?", "4")
    assert 0.0 <= score <= 1.0


def test_build_reward_model_dry_run_is_fake():
    rm = build_reward_model("Qwen/Qwen3-8B", dry_run=True)
    assert isinstance(rm, FakeRewardModel)


def test_resolve_reward_builtin_and_rm_spec():
    # Built-in heuristic.
    fn = resolve_reward("nonempty", dry_run=True)
    assert fn("p", "hello") == 1.0
    # Reward-model spec.
    rm_fn = resolve_reward("rm:Qwen/Qwen3-8B", dry_run=True)
    val = rm_fn("prompt", "a helpful answer")
    assert 0.0 <= val <= 1.0


def test_reward_model_plugs_into_rl():
    from tinker_finetune.models.schemas import RLConfig
    from tinker_finetune.training.rl_trainer import RLTrainer

    cfg = RLConfig(base_model="Qwen/Qwen3-8B", iterations=2, group_size=3,
                   prompts_per_batch=2, max_new_tokens=8, lora=LoRAConfig(rank=8))
    backend = FakeTinkerBackend(cfg.base_model, cfg.lora)
    reward_fn = resolve_reward("rm:Qwen/Qwen3-8B", dry_run=True)
    trainer = RLTrainer(backend, ByteTokenizer(), cfg, reward_fn)
    history = trainer.train(["prompt a", "prompt b"])
    assert len(history) == 2
    assert all(m.reward_mean is not None for m in history)
