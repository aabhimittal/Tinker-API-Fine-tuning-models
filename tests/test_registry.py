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


def test_inkling_is_registered_as_open_weight_flagship():
    m = get_model("thinkingmachines/Inkling")
    assert m.family == "inkling"
    assert m.license == "Apache-2.0"
    assert m.is_moe
    assert m.active_params_b < m.params_b  # 41B active of 975B
    assert m.context_length == 1_000_000
    assert m.is_multimodal
    assert "flagship" in m.tags


def test_inkling_family_variants_present():
    names = {m.name for m in list_models("inkling")}
    assert {"thinkingmachines/Inkling", "thinkingmachines/Inkling-NVFP4",
            "thinkingmachines/Inkling-Small"} <= names


def test_unknown_model_raises_with_suggestions():
    assert not is_supported("acme/closed-model")
    with pytest.raises(UnknownModelError) as exc:
        get_model("acme/closed-model")
    assert "open-weight" in str(exc.value)
