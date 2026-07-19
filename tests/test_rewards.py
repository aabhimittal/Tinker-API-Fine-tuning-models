import pytest

from tinker_finetune.rewards import get_reward_fn, list_rewards, register


def test_builtin_rewards_registered():
    names = list_rewards()
    assert {"length_target", "numeric_match", "nonempty"} <= set(names)


def test_numeric_match_reward():
    fn = get_reward_fn("numeric_match")
    assert fn("what is 6*7? 42", "the answer is 42") == 1.0
    assert fn("what is 6*7? 42", "the answer is 41") == 0.0


def test_nonempty_reward():
    fn = get_reward_fn("nonempty")
    assert fn("p", "hello") == 1.0
    assert fn("p", "   ") == 0.0


def test_unknown_reward_raises():
    with pytest.raises(KeyError):
        get_reward_fn("does_not_exist")


def test_register_custom_reward():
    @register("const_half")
    def _half(prompt, completion):
        return 0.5

    assert get_reward_fn("const_half")("a", "b") == 0.5
