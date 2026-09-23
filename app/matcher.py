"""Детерминированный пайплайн подбора: фильтры с причинами -> ранжирование -> объяснения.

Порядок шагов:
1. Пул: подрядчики нужной категории в нужном городе. Пусто -> исход no_category_in_city.
2. Жёсткие условия для каждого кандидата: дата, бюджет, формат, язык, длительность.
   Каждая причина отказа сохраняется — из них строятся объяснения пустой/неполной выдачи.
3. Ранжирование прошедших: взвешенная сумма понятных признаков, тай-брейк по id.
4. Объяснения: факты о карточке, подобранные так, чтобы отличать её от соседних.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta

from .data import CALENDAR_END, CALENDAR_START, Contractor

MAX_CARDS = 3
FILTER_ORDER = ("busy", "budget", "format", "language", "duration")

MONTHS_GEN = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля",
              "августа", "сентября", "октября", "ноября", "декабря"]

LANG_INSTR = {"русский": "на русском", "казахский": "на казахском", "английский": "на английском"}

# Лексические маркеры формата в свободном описании (корни, регистр не важен).
FORMAT_MARKERS: dict[str, list[str]] = {
    "свадьба": [r"свад", r"свадеб", r"жених", r"невест", r"бракосочет", r"молодожен", r"венча"],
    "той": [r"\bто[йяеюи]\b", r"\bтоев\b", r"ұзату", r"узату", r"беташар", r"сүндет", r"национальн", r"казахск\w* традиц"],
    "корпоратив": [r"корпоратив", r"компани", r"сотрудник", r"тимбилд", r"бренд", r"бизнес"],
    "конференция": [r"конференц", r"форум", r"саммит", r"презентац", r"делов", r"бизнес"],
    "юбилей": [r"юбиле", r"торжеств", r"годовщин"],
    "день рождения": [r"дн[яеь] рождени", r"день рождени", r"детск", r"вечеринк"],
}

WEIGHTS = {"budget": 0.40, "relevance": 0.30, "hours": 0.15, "language": 0.10, "data_quality": 0.05}


def fmt_date(d: date) -> str:
    return f"{d.day} {MONTHS_GEN[d.month - 1]}"


def fmt_kzt(v: int) -> str:
    return f"{v:,}".replace(",", " ") + " ₸"


@dataclass(frozen=True)
class MatchRequest:
    city: str
    date: date
    event_type: str
    category: str
    budget: int
    duration: int | None = None
    language: str | None = None


@dataclass
class Candidate:
    c: Contractor
    fails: list[tuple[str, str]] = field(default_factory=list)  # (код причины, текст)
    score: float = 0.0
    parts: dict[str, float] = field(default_factory=dict)
    snippet: str | None = None
    marker_hits: int = 0


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if len(p.strip()) > 15]


def _marker_hits(text: str, event_type: str) -> int:
    t = text.lower()
    return sum(len(re.findall(p, t)) for p in FORMAT_MARKERS.get(event_type, []))


def _best_snippet(desc: str, event_type: str) -> tuple[str | None, int]:
    """Самое релевантное формату предложение описания и общее число совпадений маркеров."""
    total = _marker_hits(desc, event_type)
    best, best_hits = None, 0
    for s in _sentences(desc):
        h = _marker_hits(s, event_type)
        if h > best_hits:
            best, best_hits = s, h
    if best and len(best) > 170:
        best = best[:167].rsplit(" ", 1)[0] + "…"
    return best, total


def _check(c: Contractor, req: MatchRequest) -> list[tuple[str, str]]:
    fails = []
    if req.date in c.busy:
        fails.append(("busy", f"занят {fmt_date(req.date)}"))
    if c.price > req.budget:
        fails.append(("budget", f"цена от {fmt_kzt(c.price)} выше бюджета {fmt_kzt(req.budget)}"))
    if req.event_type not in c.formats:
        fails.append(("format", f"не берёт формат «{req.event_type}»"))
    if req.language and req.language not in c.languages:
        fails.append(("language", f"не работает {LANG_INSTR.get(req.language, req.language)}"))
    if req.duration and c.max_hours is not None and c.max_hours < req.duration:
        fails.append(("duration", f"максимум {c.max_hours} ч на площадке, нужно {req.duration} ч"))
    return fails


def _score(cand: Candidate, req: MatchRequest) -> None:
    c = cand.c
    budget = max(0.0, 1.0 - c.price / req.budget)  # «цена от» — запас по бюджету ценен
    cand.snippet, cand.marker_hits = _best_snippet(c.description, req.event_type)
    specialist = 0.3 if len(c.formats) <= 2 else 0.0
    relevance = min(1.0, cand.marker_hits / 3) * 0.7 + specialist
    if req.duration and c.max_hours is not None:
        hours = min(1.0, (c.max_hours - req.duration) / 4)
    elif c.max_hours is None:
        hours = 1.0
    else:
        hours = 0.5
    language = 1.0 if req.language else len(c.languages) / 3
    data_quality = 1.0 - 0.5 * c.price_imputed - 0.5 * c.city_imputed
    cand.parts = {"budget": budget, "relevance": relevance, "hours": hours,
                  "language": language, "data_quality": data_quality}
    cand.score = round(sum(WEIGHTS[k] * v for k, v in cand.parts.items()), 6)


def _nearest_free(c: Contractor, d: date) -> date | None:
    for delta in range(1, 15):
        for cand in (d + timedelta(delta), d - timedelta(delta)):
            if CALENDAR_START <= cand <= CALENDAR_END and cand not in c.busy:
                return cand
    return None


def _comparisons(cand: Candidate, shown: list[Candidate]) -> list[str]:
    """Проверяемые отличия внутри показанной подборки в стабильном порядке."""
    c = cand.c
    others = [o for o in shown if o is not cand]
    if not others:
        return []
    facts = []
    other_languages = set().union(*(o.c.languages for o in others))
    for language in sorted(c.languages - other_languages):
        facts.append(f"единственный в подборке работает {LANG_INSTR.get(language, language)}")
    low, high = min(o.c.price for o in others), max(o.c.price for o in others)
    if c.price < low:
        facts.append(f"на {fmt_kzt(low - c.price)} дешевле ближайшего по цене варианта в подборке")
    elif c.price > high:
        facts.append(f"на {fmt_kzt(c.price - high)} дороже ближайшего по цене варианта в подборке")
    elif low < c.price < high:
        facts.append(f"на {fmt_kzt(c.price - low)} дороже самого доступного и на {fmt_kzt(high - c.price)} дешевле самого дорогого в подборке")
    elif c.price == low < high:
        facts.append(f"делит минимальную цену в подборке; на {fmt_kzt(high - c.price)} дешевле самого дорогого варианта")
    elif low < c.price == high:
        facts.append(f"на {fmt_kzt(c.price - low)} дороже самого доступного варианта в подборке")
    hours = [o.c.max_hours for o in others]
    if c.max_hours is None and all(h is not None for h in hours):
        facts.append("единственный в подборке без привязки к часам присутствия на площадке")
    elif c.max_hours is not None and all(h is not None for h in hours):
        if c.max_hours > max(hours):
            facts.append(f"до {c.max_hours} ч на площадке, у остальных — не более {max(hours)} ч")
    other_formats = set().union(*(o.c.formats for o in others))
    for event_format in sorted(c.formats - other_formats):
        facts.append(f"единственный в подборке также берёт формат «{event_format}»")
    return facts


def _description_quote(cand: Candidate, req: MatchRequest, shown: list[Candidate]) -> str | None:
    """Отличающийся фрагмент описания; приоритет у совпадения с форматом."""
    sentences = [s for s in _sentences(cand.c.description) if _marker_hits(s, req.event_type)]
    others = [o.c.description.casefold() for o in shown if o is not cand]
    sentences = sorted(enumerate(sentences), key=lambda pair: (
        -int(all(pair[1].casefold() not in desc for desc in others)),
        -_marker_hits(pair[1], req.event_type), pair[0],
    ))
    quote = sentences[0][1] if sentences else ""
    if len(quote) > 170:
        quote = quote[:170].rsplit(" ", 1)[0]
    return quote.rstrip(".!?… ") or None


def _explanation_facts(cand: Candidate, req: MatchRequest, shown: list[Candidate]) -> dict:
    c = cand.c
    return {
        "id": c.id, "price_from_kzt": c.price,
        "budget_percent": round(100 * c.price / req.budget),
        "price_imputed": c.price_imputed,
        "event_formats": sorted(c.formats), "languages": sorted(c.languages),
        "max_hours": c.max_hours, "requested_duration": req.duration,
        "description_quote": _description_quote(cand, req, shown),
        "comparisons": _comparisons(cand, shown),
    }


def _explain(cand: Candidate, req: MatchRequest, facts: dict, pool_size: int, busy_in_pool: int) -> str:
    c = cand.c
    first = f"Цена от {fmt_kzt(c.price)} — {facts['budget_percent']}% бюджета"
    if c.price_imputed:
        first += " (цена оценочная)"
    if facts["comparisons"]:
        first += "; " + "; ".join(facts["comparisons"][:2])
    second = [f"свободен {fmt_date(req.date)} (заняты {busy_in_pool} из {pool_size} в категории и городе)",
              f"берёт формат «{req.event_type}»"]
    if req.language:
        second.append(f"работает {LANG_INSTR.get(req.language, req.language)}")
    if req.duration and c.max_hours is not None:
        second.append(f"до {c.max_hours} ч при запросе на {req.duration} ч")
    elif c.max_hours is None:
        second.append("работа не привязана к часам присутствия")
    if facts["description_quote"]:
        second.append(f"в описании: «{facts['description_quote']}»")
    tail = "; ".join(second)
    return f"{first}. {tail[0].upper()}{tail[1:]}."


def _card(cand: Candidate, explanation: str) -> dict:
    c = cand.c
    return {
        "id": c.id,
        "name": c.name,
        "categories": list(c.categories),
        "city": c.city,
        "price_from_kzt": c.price,
        "languages": sorted(c.languages),
        "max_hours": c.max_hours,
        "event_formats": sorted(c.formats),
        "synthetic": c.synthetic,
        "source": c.source,
        "price_imputed": c.price_imputed,
        "city_imputed": c.city_imputed,
        "score": cand.score,
        "score_parts": {k: round(v, 3) for k, v in cand.parts.items()},
        "explanation": explanation,
        "explanation_source": "template",
    }


def _reason_summary(cands: list[Candidate], req: MatchRequest) -> str:
    counts: dict[str, int] = {}
    for cand in cands:
        for code, _ in cand.fails:
            counts[code] = counts.get(code, 0) + 1
    labels = {
        "busy": f"заняты {fmt_date(req.date)}",
        "budget": f"дороже бюджета {fmt_kzt(req.budget)}",
        "format": f"не берут формат «{req.event_type}»",
        "language": f"не работают {LANG_INSTR.get(req.language or '', req.language or '')}",
        "duration": f"не могут работать {req.duration} ч",
    }
    parts = [f"{n} {labels[k]}" for k, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    return ", ".join(parts)


def _funnel(pool: list[Candidate], req: MatchRequest) -> list[dict]:
    """Последовательное сужение пула; причины отказа при этом остаются полными."""
    steps = [{"step": "в категории и городе", "count": len(pool)}]
    labels = {
        "busy": f"свободны {fmt_date(req.date)}",
        "budget": "в бюджете",
        "format": f"берут формат «{req.event_type}»",
        "language": f"работают {LANG_INSTR.get(req.language or '', req.language or '')}",
        "duration": f"могут работать {req.duration} ч",
    }
    remaining = pool
    for code in FILTER_ORDER:
        if code == "language" and not req.language:
            continue
        if code == "duration" and req.duration is None:
            continue
        remaining = [p for p in remaining if code not in {k for k, _ in p.fails}]
        steps.append({"step": labels[code], "count": len(remaining)})
    return steps


def _hints(failed: list[Candidate], req: MatchRequest) -> list[str]:
    hints = []
    only_busy = [f for f in failed if [code for code, _ in f.fails] == ["busy"]]
    for f in sorted(only_busy, key=lambda x: x.c.id)[:2]:
        nd = _nearest_free(f.c, req.date)
        if nd:
            hints.append(f"{f.c.name} подходит по всем условиям, кроме даты: ближайший свободный день — {fmt_date(nd)}.")
    only_budget = [f for f in failed if [code for code, _ in f.fails] == ["budget"]]
    if only_budget:
        cheapest = min(only_budget, key=lambda x: (x.c.price, x.c.id))
        hints.append(f"При бюджете от {fmt_kzt(cheapest.c.price)} подошёл бы {cheapest.c.name} — он свободен и берёт этот формат.")
    return hints


def match(req: MatchRequest, catalog: list[Contractor]) -> dict:
    base = {"funnel": [], "request": {
        "city": req.city, "date": req.date.isoformat(), "event_type": req.event_type,
        "category": req.category, "budget": req.budget, "duration": req.duration, "language": req.language,
    }}

    if not (CALENDAR_START <= req.date <= CALENDAR_END):
        return {**base, "outcome": "invalid_request", "cards": [], "excluded": [], "hints": [],
                "message": f"Календари подрядчиков известны только с {fmt_date(CALENDAR_START)} по "
                           f"{fmt_date(CALENDAR_END)} 2026 — на эту дату занятость проверить нельзя."}

    pool = [Candidate(c) for c in catalog if req.category in c.categories and c.city == req.city]
    base["funnel"] = _funnel([], req)
    if not pool:
        elsewhere = sorted({c.city for c in catalog if req.category in c.categories})
        msg = f"В городе {req.city} нет подрядчиков категории «{req.category}»."
        if elsewhere:
            msg += f" Эта категория есть в: {', '.join(elsewhere)}."
        return {**base, "outcome": "no_category_in_city", "cards": [], "excluded": [], "hints": [], "message": msg}

    for cand in pool:
        cand.fails = _check(cand.c, req)
    base["funnel"] = _funnel(pool, req)
    passed = [p for p in pool if not p.fails]
    failed = [p for p in pool if p.fails]
    for p in passed:
        _score(p, req)
    passed.sort(key=lambda p: (-p.score, p.c.id))
    shown = passed[:MAX_CARDS]

    busy_in_pool = sum(1 for p in pool if req.date in p.c.busy)
    facts = {p.c.id: _explanation_facts(p, req, shown) for p in shown}
    cards = [_card(p, _explain(p, req, facts[p.c.id], len(pool), busy_in_pool)) for p in shown]
    # Не выдумываем различия между полностью совпадающими профилями.
    texts = [card["explanation"] for card in cards]
    for card in cards:
        if texts.count(card["explanation"]) > 1:
            card["explanation"] = f"Профиль {card['id']}: " + card["explanation"]
    excluded = [{"id": f.c.id, "name": f.c.name, "synthetic": f.c.synthetic,
                 "reasons": [t for _, t in f.fails]} for f in sorted(failed, key=lambda x: x.c.id)]

    pool_desc = f"Из {len(pool)} подрядчиков категории «{req.category}» в городе {req.city}"
    if not shown:
        outcome = "none_match"
        message = f"{pool_desc} ни один не проходит по условиям: {_reason_summary(failed, req)}."
    elif len(shown) < MAX_CARDS:
        outcome = "partial"
        if failed:
            message = f"{pool_desc} условиям соответствуют только {len(shown)}. Остальные: {_reason_summary(failed, req)}."
        else:
            message = (f"В городе {req.city} всего {len(pool)} подрядчик(а) категории «{req.category}» — "
                       f"все свободны и подходят, поэтому карточек меньше трёх.")
    else:
        outcome = "found"
        message = f"{pool_desc} условиям соответствуют {len(passed)}; показаны 3 лучших."

    return {**base, "outcome": outcome, "message": message, "cards": cards,
            "excluded": excluded, "hints": _hints(failed, req) if len(shown) < MAX_CARDS else []}
