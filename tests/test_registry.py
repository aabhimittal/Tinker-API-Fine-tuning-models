import pytest

from tinker_finetune.models.registry import (
    UnknownModelError,
    get_model,
    is_supported,
    list_models,
)


def test_default_model_present_and_open_weight():
    m = get_model("Qwen/Qwen3-8B")
    assert m.family == "qwen3"
    assert m.license == "Apache-2.0"
    assert m.is_dense


def test_moe_active_params_smaller_than_total():
    m = get_model("Qwen/Qwen3-30B-A3B")
    assert m.is_moe
    assert m.active_params_b < m.params_b


def test_filter_by_family():
    qwen = list_models("qwen3")
    assert qwen
    assert all(m.family == "qwen3" for m in qwen)


def test_unknown_model_raises_with_suggestions():
    assert not is_supported("acme/closed-model")
    with pytest.raises(UnknownModelError) as exc:
        get_model("acme/closed-model")
    assert "open-weight" in str(exc.value)
