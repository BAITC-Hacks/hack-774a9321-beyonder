"""HTTP API и отдача веб-страницы."""
from __future__ import annotations

from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .alternatives import alternatives
from .data import CALENDAR_END, CALENDAR_START, load_catalog
from .matcher import MatchRequest, match
from .nl_parse import parse_request

STATIC = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="Beyonder — умный подбор подрядчиков")
CATALOG = load_catalog()


class MatchIn(BaseModel):
    city: str
    date: date
    event_type: str
    category: str
    budget: int = Field(gt=0)
    duration: int | None = Field(default=None, gt=0)
    language: str | None = None


@app.get("/api/meta")
def meta() -> dict:
    return {
        "cities": sorted({c.city for c in CATALOG}),
        "categories": sorted({k for c in CATALOG for k in c.categories}),
        "event_types": sorted({f for c in CATALOG for f in c.formats}),
        "languages": sorted({l for c in CATALOG for l in c.languages}),
        "date_min": CALENDAR_START.isoformat(),
        "date_max": CALENDAR_END.isoformat(),
        "profiles": len(CATALOG),
        "synthetic_profiles": sum(c.synthetic for c in CATALOG),
    }


def _to_request(body: MatchIn) -> MatchRequest:
    known = meta()
    if body.event_type not in known["event_types"]:
        raise HTTPException(422, f"Неизвестный тип мероприятия: {body.event_type}")
    if body.language and body.language not in known["languages"]:
        raise HTTPException(422, f"Неизвестный язык: {body.language}")
    return MatchRequest(city=body.city, date=body.date, event_type=body.event_type, category=body.category,
                        budget=body.budget, duration=body.duration, language=body.language or None)


@app.post("/api/match")
def api_match(body: MatchIn) -> dict:
    return match(_to_request(body), CATALOG)


@app.post("/api/alternatives")
def api_alternatives(body: MatchIn) -> dict:
    """«План Б»: ближайшие даты (±14 дней), на которые подходит больше подрядчиков."""
    return alternatives(_to_request(body), CATALOG)


class TextIn(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


@app.post("/api/parse")
def api_parse(body: TextIn) -> dict:
    """Запрос свободным текстом -> параметры формы + что откуда взято + чего не хватает."""
    return parse_request(body.text, meta())


@app.post("/api/ask")
def api_ask(body: TextIn) -> dict:
    """Разбор текста и сразу подбор, если хватает обязательных параметров."""
    parsed = parse_request(body.text, meta())
    if parsed["missing"]:
        return {"parsed": parsed, "result": None}
    p = parsed["params"]
    return {"parsed": parsed, "result": api_match(MatchIn(**p))}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")
