"""Definition of Done на исходном датасете, без сети и LLM."""
from dataclasses import replace
from datetime import date
import re
import httpx

import pytest
from fastapi.testclient import TestClient

from app.data import CALENDAR_END, CALENDAR_START, load_catalog
from app.main import app
from app.matcher import MatchRequest, _sentences, fmt_kzt, match


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
    assert all(isinstance(c["highlights"], list) and c["highlights"]
               for c in first["cards"])
    assert len({tuple(c["highlights"]) for c in first["cards"]}) == len(first["cards"])


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
    assert "23 сентября" in first["message"]
    assert all("23 сентября" not in c["explanation"] for c in first["cards"])


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
    result = match(req, catalog)
    cards = result["cards"]
    texts = [c["explanation"] for c in cards]
    for card in cards:
        texts = [text.replace(card["name"], "") for text in texts]
    assert len(texts) == len(set(texts)) == 3
    assert all(any(ch.isdigit() for ch in text) for text in texts)
    assert all(len(text) <= 220 and not text.startswith("Цена") for text in texts)
    assert all(". Цена от " in text for text in texts)
    assert all("Свободен" not in text and "берёт формат" not in text for text in texts)
    assert "свободны" in result["message"] and "берут формат" in result["message"]
    assert any("дешевле" in text for text in texts)


def test_funnel_is_sequential_and_preserves_rejection_reasons(req, catalog):
    req = replace(req, date=date(2026, 11, 14), event_type="той",
                  budget=950_000, language="казахский", duration=6)
    response = match(req, catalog)
    assert [s["step"] for s in response["funnel"]] == [
        "в категории и городе", "свободны 14 ноября", "в бюджете",
        "берёт формат «той»", "работает на казахском", "может работать 6 ч",
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


def test_demo_messages_use_correct_agreement(catalog):
    base = MatchRequest("Алматы", date(2026, 10, 10), "свадьба", "Фотограф", 800_000)
    dense = match(base, catalog)
    rare = match(replace(base, category="Флорист"), catalog)
    none = match(replace(base, budget=1), catalog)
    assert "4 из 8 заняты" in dense["message"]
    assert "1 из 2 занят" in rare["message"]
    assert "1 берёт формат «свадьба»" == (
        f"{rare['funnel'][-1]['count']} {rare['funnel'][-1]['step']}")
    assert "4 из 8 заняты" in none["message"]


def test_venue_cards_lead_with_distinct_details(catalog):
    req = MatchRequest("Алматы", date(2026, 10, 10), "свадьба", "Банкетный зал", 5_000_000)
    cards = match(req, catalog)["cards"]
    leads = [card["explanation"].split(". Цена от ", 1)[0] for card in cards]
    assert len(leads) == len(set(leads)) == 3
    assert all(lead.startswith("В описании — «") for lead in leads)
    assert any("панорамная локация" in lead for lead in leads)
    assert any("казахской кухни" in lead for lead in leads)


def test_shared_venue_hours_move_to_message_and_capacity_leads(catalog):
    ids = {"HK-58236", "HK-90011"}
    venues = [replace(c, busy=frozenset()) for c in catalog if c.id in ids]
    req = MatchRequest("Алматы", date(2026, 10, 10), "свадьба", "Банкетный зал",
                       5_000_000, duration=6)
    result = match(req, venues)
    assert len(result["cards"]) == 2
    assert "максимум 8 ч на площадке при запросе 6 ч" in result["message"]
    assert all("8 ч" not in card["explanation"].split(". Цена от ", 1)[0]
               for card in result["cards"])
    capacity = next(card for card in result["cards"] if card["id"] == "HK-90011")
    assert "200 гостей" in capacity["explanation"].split(". Цена от ", 1)[0]
    assert "Дороже ближайшего на 200 000 ₸" in capacity["highlights"]
    assert len({tuple(c["highlights"]) for c in result["cards"]}) == 2
    assert all("8 ч" not in item for card in result["cards"] for item in card["highlights"])


@pytest.mark.parametrize("lead", ["Такой", "Этот", "Он", "Она", "Там", "Поэтому"])
def test_contextless_description_sentence_is_not_quoted(catalog, lead):
    profile = next(c for c in catalog if c.id == "HK-90011")
    profile = replace(profile, busy=frozenset(), description=(
        f"{lead} создаёт настроение на свадьбе. Более 5 лет организуем свадебные банкеты."
    ))
    req = MatchRequest("Алматы", date(2026, 10, 10), "свадьба", "Банкетный зал",
                       5_000_000)
    explanation = match(req, [profile])["cards"][0]["explanation"]
    assert "Более 5 лет организуем свадебные банкеты" in explanation
    assert f"«{lead}" not in explanation


def test_run_on_host_bio_provides_specific_quote(catalog):
    req = MatchRequest("Астана", date(2026, 11, 14), "свадьба", "Ведущий",
                       1_200_000, duration=6)
    result = match(req, catalog)
    host = next(card for card in result["cards"] if card["id"] == "HK-26808")
    assert "Сценарист команды КВН Высшей лиги" in host["explanation"]
    assert any("дешевле" in highlight.lower() for highlight in host["highlights"])
    assert len({tuple(c["highlights"]) for c in result["cards"]}) == len(result["cards"])
    assert "казахский, русский" in result["message"]


@pytest.mark.parametrize("separator", ["•", "·", ";", "\n"])
def test_description_list_separators_are_sentence_boundaries(separator):
    assert _sentences(f"Резидент клуба импровизаторов Improv Konoha {separator}Обладаю вокалом") == [
        "Резидент клуба импровизаторов Improv Konoha", "Обладаю вокалом",
    ]


def test_sanji_quote_never_contains_next_bullet_or_half_word(catalog):
    req = MatchRequest("Астана", date(2026, 11, 14), "свадьба", "Ведущий", 1_200_000)
    card = next(c for c in match(req, catalog)["cards"] if c["id"] == "HK-80581")
    quote = re.search(r"«([^»]+)»", card["explanation"]).group(1)
    profile = next(c for c in catalog if c.id == "HK-80581")
    assert quote in _sentences(profile.description)
    assert "•" not in quote and "·" not in quote and ";" not in quote
    assert "без" not in quote


def test_long_quote_is_cut_at_word_boundary_with_ellipsis(catalog):
    profile = next(c for c in catalog if c.id == "HK-80581")
    description = "Ведущий проводит свадебные церемонии " + "и работает с разными гостями " * 8
    profile = replace(profile, description=description, busy=frozenset())
    req = MatchRequest("Астана", date(2026, 11, 14), "свадьба", "Ведущий", 1_200_000)
    quote = re.search(r"«([^»]+)»", match(req, [profile])["cards"][0]["explanation"]).group(1)
    assert quote.endswith("…")
    assert len(quote) <= 100
    assert description.startswith(quote[:-1])
    assert description[len(quote) - 1] == " "


def test_highlights_are_short_comparisons_and_distinct_for_ties(catalog):
    req = MatchRequest("Алматы", date(2026, 10, 10), "свадьба", "Банкетный зал", 5_000_000)
    cards = match(req, catalog)["cards"]
    assert all(1 <= len(c["highlights"]) <= 3 for c in cards)
    assert all(len(item) <= 80 and "Из описания" not in item and "Цена от" not in item
               for c in cards for item in c["highlights"])
    assert "Единственный с казахским" in next(
        c for c in cards if c["id"] == "HK-58236"
    )["highlights"]
    assert all("той" not in item for c in cards for item in c["highlights"])
    assert len({tuple(c["highlights"]) for c in cards}) == len(cards)

    original = next(c for c in catalog if c.id == "HK-90011")
    same = replace(original, id="HK-DUP", name="Другой профиль")
    tied = match(req, [replace(original, busy=frozenset()),
                       replace(same, busy=frozenset())])["cards"]
    assert len(tied) == 2
    assert len({tuple(c["highlights"]) for c in tied}) == 2
    assert all(any("Позиция" in item for item in c["highlights"]) for c in tied)


@pytest.mark.parametrize(("profile_id", "category", "detail"), [
    ("HK-36965", "Национальный ансамбль", "Прославляем казахскую песню"),
    ("HK-57480", "Лайв-бэнд", "хитов 90-х и 2000-х"),
])
def test_music_profiles_lead_with_repertoire_not_generic_languages(
        catalog, profile_id, category, detail):
    profile = next(c for c in catalog if c.id == profile_id)
    profile = replace(profile, busy=frozenset(), formats=frozenset({"корпоратив"}))
    req = MatchRequest(profile.city, date(2026, 10, 10), "корпоратив", category, 5_000_000)
    explanation = match(req, [profile])["cards"][0]["explanation"]
    assert detail in explanation.split(". Цена от ", 1)[0]
    assert not explanation.startswith("В профиле указаны языки")


def test_numeric_rank_from_bio_beats_generic_role(catalog):
    profile = next(c for c in catalog if c.id == "HK-76268")
    profile = replace(profile, busy=frozenset(), formats=frozenset({"свадьба"}))
    req = MatchRequest("Алматы", date(2026, 12, 26), "свадьба", "Фотограф", 800_000)
    explanation = match(req, [profile])["cards"][0]["explanation"]
    assert "топ 5 Алматы" in explanation.split(". Цена от ", 1)[0]


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
