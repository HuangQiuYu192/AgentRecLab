from utils.experiment import runner


def test_source_version_tracks_nested_runtime_but_excludes_tests(tmp_path, monkeypatch):
    (tmp_path / "models").mkdir()
    runtime = tmp_path / "utils" / "experiment" / "runner.py"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("version = 1", encoding="utf-8")
    (tmp_path / "main.py").write_text("", encoding="utf-8")
    test = tmp_path / "utils" / "tests" / "test_example.py"
    test.parent.mkdir()
    test.write_text("test = 1", encoding="utf-8")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    first = runner.source_version()
    test.write_text("test = 2", encoding="utf-8")
    assert runner.source_version() == first
    runtime.write_text("version = 2", encoding="utf-8")
    assert runner.source_version() != first
