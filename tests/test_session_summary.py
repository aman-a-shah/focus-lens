"""Post-session distraction summary + heatmap (roadmap Phase 10)."""

from focuslens.app.summary import build_summary
from focuslens.pipeline import AttentionPipeline
from focuslens.session import SessionStore
from focuslens.simulate import generate_frames


def _logged_session(scenario) -> tuple[SessionStore, int]:
    store = SessionStore(":memory:")
    sid = store.start_session(0.0)
    pipe = AttentionPipeline(store=store, session_id=sid)
    for f in generate_frames(scenario, fps=30, seed=0):
        pipe.process_frame(f)
    pipe.finish()
    return store, sid


def test_heatmap_shape_and_range():
    store, sid = _logged_session([("focused", 5.0), ("distracted", 5.0)])
    summary = build_summary(store, sid, buckets=20)
    assert len(summary.heatmap) == 20
    assert all(0.0 <= v <= 1.0 for v in summary.heatmap)
    assert len(summary.ascii_heatmap()) == 20
    store.close()


def test_distracted_region_is_hotter_than_focused_region():
    store, sid = _logged_session([("focused", 6.0), ("distracted", 6.0)])
    summary = build_summary(store, sid, buckets=20)
    first_half = sum(summary.heatmap[:10]) / 10
    second_half = sum(summary.heatmap[10:]) / 10
    assert second_half > first_half  # distraction concentrates later in the session
    store.close()


def test_report_includes_states_and_intervention_feedback():
    store, sid = _logged_session([("focused", 4.0), ("fatigued", 4.0)])
    iid = store.log_intervention(sid, t=3.0, risk=12.0, fired=True)
    store.record_feedback(iid, helpful=True)

    summary = build_summary(store, sid)
    assert summary.interventions == 1 and summary.helpful == 1
    report = summary.report()
    assert "FOCUSED" in report and "distraction:" in report
    assert "1 marked helpful" in report
    store.close()


def test_empty_session_summary_is_safe():
    store = SessionStore(":memory:")
    sid = store.start_session(0.0)
    summary = build_summary(store, sid)
    assert summary.total_windows == 0 and summary.ascii_heatmap() == ""
    store.close()
