import json
import logging

from fairxai import notify as notify_mod
from fairxai.logs import get_logger


def test_notify_without_webhook_or_db(monkeypatch):
    monkeypatch.delenv("NOTIFY_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("FAIRXAI_DB_URL", raising=False)
    notify_mod.notify("info", "test", "hello")  # DB missing must not raise


def test_notify_webhook_slack_and_discord(httpserver, monkeypatch):
    monkeypatch.delenv("FAIRXAI_DB_URL", raising=False)
    monkeypatch.setenv("NOTIFY_WEBHOOK_URL", httpserver.url_for("/hook"))
    httpserver.expect_request("/hook", method="POST").respond_with_data("ok")
    monkeypatch.setenv("NOTIFY_WEBHOOK_STYLE", "slack")
    notify_mod.notify("error", "src", "boom")
    monkeypatch.setenv("NOTIFY_WEBHOOK_STYLE", "discord")
    notify_mod.notify("error", "src", "boom")
    bodies = [json.loads(r.data) for r, _ in httpserver.log]
    assert "text" in bodies[0] and "content" in bodies[1]


def test_webhook_failure_never_raises(monkeypatch):
    monkeypatch.delenv("FAIRXAI_DB_URL", raising=False)
    monkeypatch.setenv("NOTIFY_WEBHOOK_URL", "http://127.0.0.1:1/none")
    notify_mod.notify("info", "src", "x")


def test_logs_redact_keys(caplog):
    log = get_logger("fairxai.redact")
    log.propagate = True
    with caplog.at_level(logging.INFO, logger="fairxai.redact"):
        log.info("sending X-API-Key: supersecretvalue now")
    handler_filter = log.handlers[0].filters[0]
    rec = logging.LogRecord("n", logging.INFO, "", 0, "api_key=abc123 ok", (), None)
    handler_filter.filter(rec)
    assert "abc123" not in rec.msg and "***" in rec.msg
