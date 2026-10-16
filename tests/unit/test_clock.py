from datetime import timedelta

from fairxai import clock


def test_offset_applies_and_resets(tmp_path, monkeypatch):
    monkeypatch.setenv("FAIRXAI_CLOCK_OFFSET_FILE", str(tmp_path / "off"))
    assert clock.offset_seconds() == 0
    clock.advance(600)
    clock.advance(60)
    assert clock.offset_seconds() == 660
    delta = clock.now() - clock.now().replace(microsecond=0)
    assert delta < timedelta(seconds=2)
    clock.reset()
    assert clock.offset_seconds() == 0
