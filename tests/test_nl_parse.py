import pytest
from fastapi.testclient import TestClient

from app.main import app, meta
from app.nl_parse import parse_request

client = TestClient(app)


def parse(text: str) -> dict:
    return parse_request(text, meta())


@pytest.mark.parametrize("text, expected", [
    ("Нужен ведущий на казахскую свадьбу 14 ноября в Алматы, бюджет до 800 тысяч, часов на 6",
     {"city": "Алматы", "date": "2026-11-14", "event_type": "свадьба", "category": "Ведущий",
      "budget": 800_000, "duration": 6}),
    ("Я хотел бы банкетный зал в Астане на корпоратив 19.12, 3 млн тенге",
     {"city": "Астана", "date": "2026-12-19", "event_type": "корпоратив", "category": "Банкетный зал",
      "budget": 3_000_000}),
    ("фотограф на юбилей 10 октября, Алматы, 1,5 млн",
     {"category": "Фотограф", "event_type": "юбилей", "budget": 1_500_000, "date": "2026-10-10"}),
    ("ведущий церемонии на той 25 октября алматы 600к на казахском",
     {"category": "Ведущий церемонии", "event_type": "той", "budget": 600_000, "language": "казахский"}),
    ("фото и видеобудки на день рождения 7.11 Алматы 400 000 ₸ на 4 часа",
     {"category": "Фото и видеобудки", "date": "2026-11-07", "budget": 400_000, "duration": 4}),
    ("шоу-балет на корпоратив 20 декабря алматы бюджет 1.5 млн",
     {"category": "Танцевальный коллектив", "budget": 1_500_000, "date": "2026-12-20"}),
])
def test_parses_fields(text, expected):
    params = parse(text)["params"]
    for field, value in expected.items():
        assert params[field] == value, field


def test_hotel_not_detected_in_hotel_word_inside_verb():
    assert parse("Хотел бы ведущего на свадьбу")["params"]["category"] == "Ведущий"


def test_missing_fields_reported_and_second_category_noted():
    r = parse("нужен ведущий и фотограф на свадьбу")
    assert r["missing"] == ["city", "date", "budget"]
    assert any("Фотограф" in n for n in r["notes"])


def test_kazakh_wedding_suggests_language_and_toi_without_guessing():
    r = parse("Нужен ведущий на казахскую свадьбу 14 ноября в Алматы до 800 тысяч")
    assert r["params"]["event_type"] == "свадьба" and r["params"]["language"] is None
    note = next(n for n in r["notes"] if "казахск" in n)
    assert "«казахский»" in note and "«той»" in note


def test_date_outside_calendar_is_flagged():
    r = parse("лайв-бэнд на свадьбу 5 января в Алматы 2 млн")
    assert any("вне календаря" in n for n in r["notes"])


def test_every_found_field_has_source_fragment():
    r = parse("Нужен ведущий на свадьбу 14 ноября в Алматы до 800 тысяч")
    assert r["found"] and all(f["fragment"] for f in r["found"])


def test_ask_runs_match_when_complete_and_is_deterministic():
    text = "Нужен ведущий на свадьбу 14 ноября в Алматы, бюджет до 800 тысяч"
    a = client.post("/api/ask", json={"text": text}).json()
    b = client.post("/api/ask", json={"text": text}).json()
    assert a["result"]["outcome"] in {"found", "partial", "none_match"}
    assert [c["id"] for c in a["result"]["cards"]] == [c["id"] for c in b["result"]["cards"]]


def test_ask_without_required_fields_returns_no_result():
    r = client.post("/api/ask", json={"text": "нужен фотограф"}).json()
    assert r["result"] is None and "city" in r["parsed"]["missing"]
