from datetime import date

from fastapi.testclient import TestClient

from app.data import CALENDAR_END, CALENDAR_START
from app.main import app

client = TestClient(app)

PHOTO = {"city": "Алматы", "event_type": "свадьба", "category": "Фотограф", "budget": 800000}


def alt(**q) -> dict:
    r = client.post("/api/alternatives", json=q)
    assert r.status_code == 200
    return r.json()


def match(**q) -> dict:
    return client.post("/api/match", json=q).json()


def test_busy_date_offers_nearest_dates_with_bigger_selection():
    r = alt(**PHOTO, date="2026-12-12")
    assert r["base"]["passed"] == 1 and r["options"]
    for o in r["options"]:
        assert o["passed"] > r["base"]["passed"]
        assert 1 <= abs(o["delta_days"]) <= 14
        assert CALENDAR_START <= date.fromisoformat(o["date"]) <= CALENDAR_END
    assert "Ближайшие даты" in r["message"]


def test_offered_date_really_gives_that_selection_in_main_match():
    first = alt(**PHOTO, date="2026-12-12")["options"][0]
    real = match(**PHOTO, date=first["date"])
    assert [c["name"] for c in real["cards"]] == first["top"]


def test_nearest_full_selection_comes_first_and_order_is_deterministic():
    a = alt(**PHOTO, date="2026-12-12")
    b = alt(**PHOTO, date="2026-12-12")
    assert a == b
    deltas = [abs(o["delta_days"]) for o in a["options"] if o["passed"] >= 3]
    assert deltas == sorted(deltas)


def test_already_full_selection_needs_no_plan_b():
    r = alt(**PHOTO, date="2026-10-10")
    assert r["options"] == [] and "лучшая возможная" in r["message"]


def test_when_date_is_not_the_problem_it_says_so_with_reasons():
    r = alt(city="Алматы", date="2026-10-10", event_type="свадьба", category="Ведущий", budget=1)
    assert r["options"] == []
    assert "Дело не в дате" in r["message"] and "дороже бюджета" in r["message"]


def test_missing_category_in_city():
    r = alt(city="Астана", date="2026-11-14", event_type="той", category="Лайв-бэнд", budget=500000)
    assert r["options"] == [] and "нет категории" in r["message"]


def test_unknown_event_type_is_422_like_match():
    r = client.post("/api/alternatives", json={**PHOTO, "event_type": "пикник", "date": "2026-10-10"})
    assert r.status_code == 422
