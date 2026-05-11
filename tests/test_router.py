import pytest

from xau_pro_bot.signals.router import StreamRouter


def test_router_returns_list(all_tfs):
    results = StreamRouter().analyze(all_tfs)
    assert isinstance(results, list)
    for sig in results:
        assert sig.get("stream") in ("intraday", "swing", "scalp")


def test_router_continues_on_analyzer_exception(all_tfs, monkeypatch):
    router = StreamRouter()

    def boom(_data):
        raise RuntimeError("scalp blew up")

    monkeypatch.setattr(router.analyzers["scalp"], "analyze", boom)
    results = router.analyze(all_tfs)
    assert isinstance(results, list)
