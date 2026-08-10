from types import SimpleNamespace

from text_render_protocol_predictor.training.sampler import MaskSourceBalancedSampler


def test_mask_source_sampler_is_balanced_and_deterministic() -> None:
    entries = [
        SimpleNamespace(mask_supervision=SimpleNamespace(source="files")),
        SimpleNamespace(mask_supervision=SimpleNamespace(source="geometry")),
        SimpleNamespace(mask_supervision=None),
        SimpleNamespace(mask_supervision=None),
        SimpleNamespace(mask_supervision=None),
    ]
    dataset = SimpleNamespace(entries=entries, __len__=lambda self: len(entries))
    dataset = type("Dataset", (), {"entries": entries, "__len__": lambda self: 10})()
    sampler = MaskSourceBalancedSampler(
        dataset,
        weights={"files": 0.4, "geometry": 0.4, "none": 0.2},
        seed=7,
    )

    first = list(sampler)
    assert first == list(sampler)
    sources = [
        entries[index].mask_supervision.source
        if entries[index].mask_supervision is not None
        else "none"
        for index in first
    ]
    assert sources.count("files") == 4
    assert sources.count("geometry") == 4
    assert sources.count("none") == 2
    sampler.set_epoch(1)
    assert list(sampler) != first
