import threading
from datetime import date
from types import SimpleNamespace

import pytest
from src.services import quota, router

@pytest.fixture(autouse=True)
def small_limits(tmp_path, monkeypatch):
    # use a throwaway database and small limits in every test
    monkeypatch.setattr(quota, "USAGE_DB_PATH", tmp_path / "usage.db")
    monkeypatch.setattr(quota, "REMOTE_LIMIT_PER_IP", 3)
    monkeypatch.setattr(quota, "REMOTE_LIMIT_TOTAL", 5)

def test_limit_per_ip():
    assert [quota.try_use_remote("1.1.1.1") for _ in range(4)] == [True, True, True, False]
    # another visitor still has their own calls
    assert quota.try_use_remote("2.2.2.2")

def test_new_day_resets(monkeypatch):
    for _ in range(3):
        quota.try_use_remote("1.1.1.1")
    assert not quota.try_use_remote("1.1.1.1")
    tomorrow = SimpleNamespace(today=lambda: date(2099, 1, 1))
    monkeypatch.setattr(quota, "date", tomorrow)
    assert quota.try_use_remote("1.1.1.1")

def test_many_request_at_once():
    # 25 threads hit the same IP together and only 3 can pass
    result = []
    threads = [threading.Thread(target=lambda: result.append(quota.try_use_remote("1.1.1.1"))) for _ in range(50)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert result.count(True) == 3

# fake a model so no credits are spent and model loaded when testing
def fake_model(route_name, fail=False):
    def critique(*args):
        if fail:
            raise RuntimeError(f"{route_name} broke")
        return router.Critique(score=50, evaluation="- tip", model_name=route_name, route=route_name)
    return critique


@pytest.fixture
def fakes(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "photo.jpg").write_bytes(b"x" * 100)
    monkeypatch.setattr(router, "get_token", lambda: "hf_team_token")
    monkeypatch.setattr(router, "local_critique", fake_model("Local"))
    monkeypatch.setattr(router, "remote_critique", fake_model("Remote"))
    monkeypatch.setattr(router.gr, "Warning", lambda message: None)  # no UI here


def visit(ip, use_local=False):
    request = router.gr.Request(client={"host": ip}, headers={})
    _, status = router.score_artwork("photo.jpg", "Composition", 0.0, 1.0, use_local, request)
    return status


def test_router_remote_until_limit(fakes):
    statuses = [visit("1.1.1.1") for _ in range(4)]
    assert ["Remote" in s for s in statuses] == [True, True, True, False]
    assert "daily limit reached" in statuses[-1]


def test_router_local_choice_costs_nothing(fakes):
    for _ in range(10):
        assert "Local" in visit("1.1.1.1", use_local=True)
    assert "Remote" in visit("1.1.1.1")  # remote calls still all available


def test_router_remote_fails_goes_local(fakes, monkeypatch):
    monkeypatch.setattr(router, "remote_critique", fake_model("Remote", fail=True))
    assert "remote unavailable" in visit("1.1.1.1")


def test_router_no_token_goes_local(fakes, monkeypatch):
    monkeypatch.setattr(router, "get_token", lambda: None)
    assert "no credentials" in visit("1.1.1.1")


def test_router_no_photo(fakes):
    with pytest.raises(router.gr.Error):
        router.score_artwork(None, "Composition", 0.0, 1.0, False, router.gr.Request())

def test_router_quota_per_visitor_behind_proxy(fakes):
    def via_proxy(visitor):
        request = router.gr.Request(client={"host": "172.30.19.20"}, headers={"X-Forwarded-For": f"{visitor}, 172.30.19.20"})
        _, status = router.score_artwork("photo.jpg", "Composition", 0.0, 1.0, False, request)
        return status
    for _ in range(3):
        via_proxy("203.0.113.7")
    assert "Remote" in via_proxy("198.51.100.4")

def test_failover_counted_under_the_model_that_answered(fakes):
    # if daily limit is reached, the failover to local model should be counted under "local"
    # and failover under reason="daily_limit"
    from prometheus_client import REGISTRY
    def value(name, **labels):
        return REGISTRY.get_sample_value(name, labels) or 0
    before = (value("photo_critic_requests_total", model="local", outcome="success"),
              value("photo_critic_failovers_total", reason="daily_limit"))
    for _ in range(4):
        visit("1.1.1.1")
    after = (value("photo_critic_requests_total", model="local", outcome="success"),
             value("photo_critic_failovers_total", reason="daily_limit"))
    # compare the change
    assert (after[0] - before[0], after[1] - before[1]) == (1, 1)