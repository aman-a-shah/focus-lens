import textwrap

from focuslens.config import Config, _deep_merge, load_config


def test_defaults_load_from_bundled_yaml():
    cfg = load_config()
    assert isinstance(cfg, Config)
    assert cfg.capture.target_fps == 30
    assert cfg.face_mesh.refine_landmarks is True
    assert cfg.capture.buffer_seconds == 5.0


def test_deep_merge_overrides_nested_scalars_only():
    base = {"capture": {"camera_index": 0, "width": 1280}}
    override = {"capture": {"camera_index": 2}}
    merged = _deep_merge(base, override)
    assert merged == {"capture": {"camera_index": 2, "width": 1280}}


def test_override_file_is_merged(tmp_path):
    override = tmp_path / "over.yaml"
    override.write_text(textwrap.dedent("""
            capture:
              camera_index: 3
            logging:
              level: DEBUG
            """))
    cfg = load_config(override)
    assert cfg.capture.camera_index == 3
    assert cfg.logging.level == "DEBUG"
    # untouched defaults survive
    assert cfg.capture.target_fps == 30
