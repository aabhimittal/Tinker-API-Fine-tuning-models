"""Edge cases for deterministic sharding, resumable cursors and mixture sampling."""

from __future__ import annotations

import json

import pytest

from tinker_finetune.data.sharding import (
    Cursor,
    ExhaustionPolicy,
    MixtureSampler,
    MixtureSource,
    ReshardError,
    ShardedStream,
    ShardSpec,
    stable_hash,
)

ROWS = [f"row-{i:05d}" for i in range(500)]


def specs(world_size: int, *, buckets: int = 256, seed: int = 7) -> list[ShardSpec]:
    return [ShardSpec(rank=r, world_size=world_size, num_buckets=buckets, seed=seed) for r in range(world_size)]


def test_stable_hash_is_stable_and_seed_sensitive():
    assert stable_hash("abc") == stable_hash("abc")
    assert stable_hash("abc") != stable_hash("abc", seed=1)
    # Non-string keys must not raise and must stay distinct.
    assert stable_hash({"a": 1}) != stable_hash({"a": 2})
    assert stable_hash(b"abc") == stable_hash("abc")


def test_every_row_lands_on_exactly_one_rank():
    assignment = [[row for row in ROWS if spec.owns(row)] for spec in specs(4)]
    flat = [row for shard in assignment for row in shard]
    assert sorted(flat) == sorted(ROWS)
    assert len(set(flat)) == len(ROWS)


def test_shards_are_roughly_balanced():
    sizes = [sum(1 for row in ROWS if spec.owns(row)) for spec in specs(4)]
    assert min(sizes) > 0.5 * (len(ROWS) / 4)


def test_growing_the_corpus_does_not_move_existing_rows():
    spec = ShardSpec(rank=1, world_size=4, num_buckets=256)
    before = {row: spec.owns(row) for row in ROWS}
    grown = ROWS + [f"new-{i}" for i in range(200)]
    after = {row: spec.owns(row) for row in grown if row in before}
    assert before == after


def test_adding_a_worker_moves_only_a_small_share():
    """Rendezvous placement: W -> W+1 should move ~1/(W+1), not ~everything."""
    old = {row: next(s.rank for s in specs(4) if s.owns(row)) for row in ROWS}
    new = {row: next(s.rank for s in specs(5) if s.owns(row)) for row in ROWS}
    moved = sum(1 for row in ROWS if old[row] != new[row])
    assert moved / len(ROWS) < 0.40  # modulo sharding would move ~80%


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rank": 4, "world_size": 4},
        {"rank": -1, "world_size": 4},
        {"rank": 0, "world_size": 0},
        {"rank": 0, "world_size": 8, "num_buckets": 4},
    ],
)
def test_invalid_shard_specs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ShardSpec(**kwargs)


def test_single_worker_owns_everything():
    spec = ShardSpec(rank=0, world_size=1, num_buckets=1)
    assert all(spec.owns(row) for row in ROWS)
    assert spec.buckets() == (0,)


def test_stream_covers_the_shard_exactly_once_per_epoch():
    spec = specs(3)[0]
    stream = ShardedStream(ROWS, spec, seed=3)
    seen = list(stream)
    owned = {ROWS[i] for i in stream.owned_indices}
    assert sorted(seen) == sorted(owned)
    assert len(seen) == len(stream)


def test_resume_from_cursor_has_no_gap_and_no_duplicate():
    stream = ShardedStream(ROWS, specs(3)[1], seed=11)
    pairs = list(stream.iter_from())
    cut = len(pairs) // 3
    cursor = pairs[cut - 1][1]
    resumed = [row for row, _ in stream.iter_from(cursor)]
    assert [row for row, _ in pairs[:cut]] + resumed == [row for row, _ in pairs]


def test_resume_past_the_end_yields_nothing():
    stream = ShardedStream(ROWS, specs(3)[0], seed=1)
    last_cursor = list(stream.iter_from())[-1][1]
    assert list(stream.iter_from(last_cursor)) == []


def test_epochs_reshuffle_but_replay_identically():
    stream = ShardedStream(ROWS, specs(2)[0], seed=5, max_epochs=2)
    e0, e1 = stream.epoch_order(0), stream.epoch_order(1)
    assert sorted(e0) == sorted(e1)
    assert e0 != e1
    assert stream.epoch_order(1) == e1  # deterministic replay


def test_shuffle_off_keeps_corpus_order():
    spec = specs(2)[0]
    stream = ShardedStream(ROWS, spec, shuffle=False)
    assert stream.epoch_order(0) == stream.owned_indices == sorted(stream.owned_indices)


def test_empty_shard_terminates_even_when_unbounded():
    """An infinite stream over an empty shard must stop, not spin."""
    spec = ShardSpec(rank=0, world_size=2, num_buckets=2)
    stream = ShardedStream([], spec, max_epochs=None)
    assert list(stream.iter_from()) == []


def test_more_shards_than_rows_leaves_some_workers_empty_but_valid():
    rows = ["a", "b"]
    streams = [ShardedStream(rows, spec) for spec in specs(8, buckets=64)]
    assert sum(len(s) for s in streams) == 2


def test_max_epochs_zero_yields_nothing():
    assert list(ShardedStream(ROWS, specs(2)[0], max_epochs=0)) == []


def test_negative_max_epochs_rejected():
    with pytest.raises(ValueError):
        ShardedStream(ROWS, specs(2)[0], max_epochs=-1)


def test_cursor_round_trips_through_json():
    cursor = Cursor(epoch=2, index=17, world_size=4, rank=3, num_buckets=256)
    assert Cursor.from_dict(json.loads(json.dumps(cursor.to_dict()))) == cursor


def test_cursor_from_dict_ignores_unknown_keys():
    assert Cursor.from_dict({"epoch": 1, "index": 2, "job_id": 9}).epoch == 1


def test_negative_cursor_rejected():
    with pytest.raises(ValueError):
        Cursor(epoch=-1)


def test_resume_under_a_different_topology_is_reported():
    stream = ShardedStream(ROWS, specs(3)[0], seed=2, max_epochs=2)
    stale = Cursor(epoch=1, index=4, world_size=4, rank=0, num_buckets=256)
    with pytest.raises(ReshardError):
        list(stream.iter_from(stale))
    resumed = list(stream.iter_from(stale, allow_reshard=True))
    assert len(resumed) == len(stream)  # restarts the epoch rather than skipping rows
    assert resumed[0][1].epoch == 1


def test_key_fn_shards_on_a_field_not_the_whole_row():
    rows = [{"id": f"user-{i}", "text": f"payload {i}"} for i in range(200)]
    spec = ShardSpec(rank=0, world_size=4, num_buckets=64)
    by_id = ShardedStream(rows, spec, key_fn=lambda r: r["id"])
    # The same user is always on the same shard even if the text changes.
    mutated = [{"id": r["id"], "text": r["text"] + "!"} for r in rows]
    assert by_id.owned_indices == ShardedStream(mutated, spec, key_fn=lambda r: r["id"]).owned_indices


# --- mixture sampling ---------------------------------------------------------


def sources(**weights: float) -> list[MixtureSource[str]]:
    return [MixtureSource(name=n, rows=[f"{n}-{i}" for i in range(50)], weight=w) for n, w in weights.items()]


def test_temperature_flattens_the_mixture():
    sharp = MixtureSampler(sources(big=0.9, small=0.1), temperature=1.0).probabilities()
    flat = MixtureSampler(sources(big=0.9, small=0.1), temperature=4.0).probabilities()
    assert sharp["small"] < flat["small"] < 0.5


def test_zero_temperature_collapses_to_the_heaviest_source():
    probs = MixtureSampler(sources(a=0.6, b=0.4), temperature=0.0).probabilities()
    assert probs == {"a": 1.0}


def test_realized_proportions_track_the_target():
    sampler = MixtureSampler(sources(a=0.75, b=0.25), seed=4)
    sampler.take(2000)
    realized = sampler.realized_proportions()
    assert abs(realized["a"] - 0.75) < 0.05


def test_sampling_is_deterministic_for_a_seed():
    a = MixtureSampler(sources(a=1, b=1), seed=9).take(30)
    b = MixtureSampler(sources(a=1, b=1), seed=9).take(30)
    c = MixtureSampler(sources(a=1, b=1), seed=10).take(30)
    assert a == b and a != c


def test_zero_weight_source_is_never_sampled():
    sampler = MixtureSampler(sources(a=1.0, b=0.0), seed=1)
    assert {name for name, _ in sampler.take(100)} == {"a"}


def test_cycle_policy_upsamples_a_small_source():
    small = MixtureSource(name="small", rows=["only"], weight=1.0)
    big = MixtureSource(name="big", rows=[f"b{i}" for i in range(100)], weight=1.0)
    drawn = MixtureSampler([small, big], seed=2, policy=ExhaustionPolicy.cycle).take(60)
    assert sum(1 for name, _ in drawn if name == "small") > 1


def test_drain_policy_renormalizes_onto_the_survivor():
    small = MixtureSource(name="small", rows=["only"], weight=1.0)
    big = MixtureSource(name="big", rows=[f"b{i}" for i in range(100)], weight=1.0)
    drawn = MixtureSampler([small, big], seed=2, policy=ExhaustionPolicy.drain).take(60)
    assert sum(1 for name, _ in drawn if name == "small") == 1
    assert len(drawn) == 60


def test_stop_policy_ends_the_mixture_at_first_exhaustion():
    small = MixtureSource(name="small", rows=["only"], weight=1.0)
    big = MixtureSource(name="big", rows=[f"b{i}" for i in range(100)], weight=1.0)
    drawn = MixtureSampler([small, big], seed=2, policy=ExhaustionPolicy.stop).take(200)
    assert len(drawn) < 200


def test_all_sources_empty_yields_nothing():
    empty = [MixtureSource(name="a", rows=[], weight=1.0)]
    assert MixtureSampler(empty).take(10) == []


def test_take_zero_is_a_no_op():
    sampler = MixtureSampler(sources(a=1))
    assert sampler.take(0) == []
    assert sampler.realized_proportions() == {"a": 0.0}


@pytest.mark.parametrize("n", [-1])
def test_negative_take_rejected(n):
    with pytest.raises(ValueError):
        MixtureSampler(sources(a=1)).take(n)


def test_invalid_mixtures_are_rejected():
    with pytest.raises(ValueError):
        MixtureSampler([])
    with pytest.raises(ValueError):
        MixtureSampler(sources(a=0.0, b=0.0))
    with pytest.raises(ValueError):
        MixtureSampler(sources(a=1.0), temperature=-1.0)
    with pytest.raises(ValueError):
        MixtureSource(name="a", rows=[], weight=float("nan"))
    with pytest.raises(ValueError):
        MixtureSource(name="", rows=[])
    dupe = [MixtureSource(name="a", rows=["x"]), MixtureSource(name="a", rows=["y"])]
    with pytest.raises(ValueError):
        MixtureSampler(dupe)


def test_extreme_temperature_does_not_produce_nan_probabilities():
    probs = MixtureSampler(sources(a=0.9, b=0.1), temperature=1e-6).probabilities()
    assert abs(sum(probs.values()) - 1.0) < 1e-9
