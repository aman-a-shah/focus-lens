"""Window-sequence datasets for PersonalFocusNet (roadmap Phase 7)."""

import torch

from focuslens.focusnet.dataset import (
    FeatureNormalizer,
    SyntheticFocusDataset,
    WindowSequenceDataset,
)
from focuslens.labeling import label_all_sessions
from focuslens.pipeline import AttentionPipeline
from focuslens.session import SessionStore
from focuslens.simulate import DEFAULT_SCENARIO, generate_frames
from focuslens.states import NUM_STATES
from focuslens.window import NUM_FEATURES


def test_synthetic_dataset_shapes_and_label_coverage():
    ds = SyntheticFocusDataset(n_per_class=20, seq_len=12, seed=0)
    assert len(ds) == 20 * NUM_STATES
    item = ds[0]
    assert item["seq"].shape == (12, NUM_FEATURES)
    assert set(int(y) for y in ds.labels.tolist()) == set(range(NUM_STATES))


def test_normalizer_standardizes_and_roundtrips():
    x = torch.randn(50, 8, NUM_FEATURES) * 5 + 3
    norm = FeatureNormalizer.fit(x)
    z = norm.apply(x)
    assert abs(float(z.mean())) < 0.1 and abs(float(z.std()) - 1.0) < 0.1
    restored = FeatureNormalizer.from_state_dict(norm.state_dict())
    assert torch.allclose(restored.apply(x), z, atol=1e-5)


def test_from_store_builds_left_padded_sequences():
    store = SessionStore(":memory:")
    sid = store.start_session(0.0)
    pipe = AttentionPipeline(store=store, session_id=sid)
    frames = generate_frames(DEFAULT_SCENARIO, fps=30, seed=0)
    for f in frames:
        pipe.process_frame(f)
    pipe.finish()
    label_all_sessions(store)

    ds = WindowSequenceDataset.from_store(store, seq_len=30)
    assert len(ds) == store.window_count(sid)
    assert ds[0]["seq"].shape == (30, NUM_FEATURES)
    # First labelled window has only itself -> the rest of the sequence is zero-padding.
    assert float(ds.raw[0][:29].abs().sum()) == 0.0
    store.close()
