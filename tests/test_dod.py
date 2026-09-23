"""Definition of Done на исходном датасете, без сети и LLM."""
from dataclasses import replace
from datetime import date
import httpx

import pytest
from fastapi.testclient import TestClient

from app.data import CALENDAR_END, CALENDAR_START, load_catalog
from app.main import app
from app.matcher import MatchRequest, fmt_kzt, match


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("EXPLANATIONS_LLM_ENABLED", "false")

    def forbidden(*args, **kwargs):
        raise AssertionError("Тесты DoD не должны обращаться к сети")

    async def forbidden_async(*args, **kwargs):
        forbidden()

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", forbidden_async)


@pytest.fixture(scope="module")
def catalog():
    return [c for c in load_catalog() if c.source == "organizers"]


@pytest.fixture
def req():
    return MatchRequest("Алматы", date(2026, 9, 23), "корпоратив", "Ведущий", 2_000_000)


def test_determinism_and_card_contract(req, catalog):
    first, second = match(req, catalog), match(req, catalog)
    assert len(catalog) == 66
    assert first == second
    old_fields = {"id", "name", "categories", "city", "price_from_kzt", "languages",
                  "max_hours", "event_formats", "synthetic", "source", "price_imputed",
                  "city_imputed", "score", "score_parts", "explanation"}
    assert all(old_fields <= c.keys() for c in first["cards"])
    assert all(c["explanation_source"] == "template" for c in first["cards"])


def test_busy_contractors_never_shown(req, catalog):
    for day in (date(2026, 9, 23), date(2026, 11, 14), date(2026, 12, 26)):
        response = match(replace(req, date=day), catalog)
        shown = {c["id"] for c in response["cards"]}
        assert not any(c.id in shown and day in c.busy for c in catalog)


def test_two_dates_change_results_with_busy_reason(req, catalog):
    first = match(req, catalog)
    second = match(replace(req, date=date(2026, 9, 24)), catalog)
    assert [c["id"] for c in first["cards"]] != [c["id"] for c in second["cards"]]
    assert "HK-27222" in {c["id"] for c in first["cards"]}
    excluded = next(c for c in second["excluded"] if c["id"] == "HK-27222")
    assert "занят 24 сентября" in excluded["reasons"]
    assert all("23 сентября" in c["explanation"] for c in first["cards"])


@pytest.mark.parametrize(("changes", "outcome", "count"), [
    ({"category": "Лайв-бэнд", "city": "Астана"}, "no_category_in_city", 0),
    ({"budget": 1}, "none_match", 0),
    ({"category": "Флорист", "event_type": "свадьба"}, "partial", 2),
    ({}, "found", 3),
])
def test_outcomes(req, catalog, changes, outcome, count):
    response = match(replace(req, **changes), catalog)
    assert response["outcome"] == outcome
    assert len(response["cards"]) == count
    assert response["message"].strip()


def test_rare_category_explains_partial_result(req, catalog):
    response = match(replace(req, category="Флорист", event_type="свадьба"), catalog)
    assert 0 < len(response["cards"]) <= 3
    assert response["outcome"] == "partial"
    assert "меньше трёх" in response["message"] or "только" in response["message"]
    assert "Флорист" in response["message"]


def test_budget_hint_refers_to_profile_without_guessing_gender(req, catalog):
    shown_id = match(req, catalog)["cards"][0]["id"]
    contractor = next(c for c in catalog if c.id == shown_id)
    contractor = replace(contractor, name="Софи Хаттер", price=req.budget + 1)
    response = match(req, [contractor])
    assert response["hints"] == [
        f"При бюджете от {fmt_kzt(contractor.price)} подошёл бы профиль «Софи Хаттер»: "
        "свободен и берёт этот формат.",
    ]


@pytest.mark.parametrize("day", [date(2026, 9, 22), date(2027, 1, 1)])
def test_out_of_calendar_is_invalid(req, catalog, day):
    response = match(replace(req, date=day), catalog)
    assert response["outcome"] == "invalid_request"
    assert response["cards"] == response["funnel"] == []
    assert "занятость проверить нельзя" in response["message"]


@pytest.mark.parametrize("day", [CALENDAR_START, CALENDAR_END])
def test_calendar_boundaries_are_inclusive(req, catalog, day):
    assert match(replace(req, date=day), catalog)["outcome"] != "invalid_request"


def test_explanations_remain_distinct_without_names(req, catalog):
    cards = match(req, catalog)["cards"]
    texts = [c["explanation"] for c in cards]
    for card in cards:
        texts = [text.replace(card["name"], "") for text in texts]
    assert len(texts) == len(set(texts)) == 3
    assert all(any(ch.isdigit() for ch in text) for text in texts)
    assert any("единственный" in text for text in texts)
    assert any("дешевле" in text for text in texts)


def test_funnel_is_sequential_and_preserves_rejection_reasons(req, catalog):
    req = replace(req, date=date(2026, 11, 14), event_type="той",
                  budget=950_000, language="казахский", duration=6)
    response = match(req, catalog)
    assert [s["step"] for s in response["funnel"]] == [
        "в категории и городе", "свободны 14 ноября", "в бюджете",
        "берут формат «той»", "работают на казахском", "могут работать 6 ч",
    ]
    remaining = [c for c in catalog if c.city == req.city and req.category in c.categories]
    counts = [len(remaining)]
    for predicate in (
        lambda c: req.date not in c.busy,
        lambda c: c.price <= req.budget,
        lambda c: req.event_type in c.formats,
        lambda c: req.language in c.languages,
        lambda c: c.max_hours is None or c.max_hours >= req.duration,
    ):
        remaining = [c for c in remaining if predicate(c)]
        counts.append(len(remaining))
    assert [s["count"] for s in response["funnel"]] == counts
    assert len(response["cards"]) == min(3, counts[-1])
    assert any(len(c["reasons"]) > 1 for c in response["excluded"])


def test_http_contract():
    with TestClient(app) as client:
        assert client.get("/api/meta").json()["profiles"] >= 66
        payload = {"city": "Алматы", "date": "2026-09-23", "event_type": "корпоратив",
                   "category": "Ведущий", "budget": 2_000_000}
        response = client.post("/api/match", json=payload)
        assert response.status_code == 200
        result = response.json()
        assert {"request", "outcome", "message", "cards", "excluded", "hints", "funnel"} <= result.keys()
        assert all(c["explanation_source"] == "template" for c in result["cards"])
        assert client.post("/api/match", json={**payload, "date": "2027-01-01"}).json()["outcome"] == "invalid_request"
