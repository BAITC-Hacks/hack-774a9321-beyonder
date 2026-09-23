"""«План Б»: если на выбранную дату подходящих мало, найти ближайшие даты с лучшей подборкой.

Шаг поверх того же пайплайна: для соседних дат (±14 дней в пределах календаря) прогоняются
те же жёсткие фильтры и тот же рейтинг, что и в основном подборе. Объяснения не генерируются —
считается, сколько подрядчиков прошло бы и кто вошёл бы в тройку. Если мешает не дата
(бюджет, формат, язык, длительность), сервис прямо говорит, что смена даты не поможет.
Детерминировано и без сети.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta

from .data import CALENDAR_END, CALENDAR_START, Contractor
from .matcher import MAX_CARDS, Candidate, MatchRequest, _check, _plural, _reason_summary, _score, fmt_date

WINDOW_DAYS = 14
MAX_OPTIONS = 3
WEEKDAYS = ["пн", "вт", "ср", "чт", "пт", "сб", "вс"]


def _passed(req: MatchRequest, pool: list[Contractor]) -> list[Candidate]:
    passed = [Candidate(c) for c in pool if not _check(c, req)]
    for p in passed:
        _score(p, req)
    passed.sort(key=lambda p: (-p.score, p.c.id))
    return passed


def _variants(n: int) -> str:
    return f"{n} {_plural(n, 'вариант', 'варианта', 'вариантов')}"


def alternatives(req: MatchRequest, catalog: list[Contractor]) -> dict:
    pool = [c for c in catalog if req.category in c.categories and c.city == req.city]
    base_passed = _passed(req, pool) if CALENDAR_START <= req.date <= CALENDAR_END else []
    base = {"date": req.date.isoformat(), "label": fmt_date(req.date), "passed": len(base_passed)}

    def result(message: str, options: list[dict] | None = None) -> dict:
        return {"base": base, "options": options or [], "message": message}

    if not pool:
        return result(f"В городе {req.city} нет категории «{req.category}» — другая дата не поможет.")
    if not (CALENDAR_START <= req.date <= CALENDAR_END):
        return result(f"Календари известны только с {fmt_date(CALENDAR_START)} по {fmt_date(CALENDAR_END)} "
                      f"{CALENDAR_END.year} — выберите дату в этом окне.")

    # Сколько подрядчиков вообще может пройти, если подобрать удачную дату.
    reachable = [c for c in pool if all(code == "busy" for code, _ in _check(c, req))]
    best_possible = min(MAX_CARDS, len(reachable))
    if best_possible == 0:
        blockers = _reason_summary([Candidate(c, fails=_check(c, req)) for c in pool], req)
        return result(f"Дело не в дате: из {len(pool)} подрядчиков категории: {blockers}. "
                      f"Смена даты не поможет — измените эти условия.")
    if len(base_passed) >= best_possible:
        return result(f"На {fmt_date(req.date)} уже лучшая возможная подборка — "
                      f"{_variants(min(len(base_passed), MAX_CARDS))}.")

    found = []
    for delta in range(1, WINDOW_DAYS + 1):
        for d in (req.date - timedelta(delta), req.date + timedelta(delta)):
            if not (CALENDAR_START <= d <= CALENDAR_END):
                continue
            passed = _passed(replace(req, date=d), reachable)
            if len(passed) > len(base_passed):
                found.append((d, d - req.date, passed))
    found.sort(key=lambda x: (-min(len(x[2]), MAX_CARDS), abs(x[1].days), x[0]))

    options = [{
        "date": d.isoformat(),
        "label": fmt_date(d),
        "weekday": WEEKDAYS[d.weekday()],
        "weekend": d.weekday() >= 5,
        "delta_days": delta.days,
        "passed": len(passed),
        "top": [p.c.name for p in passed[:MAX_CARDS]],
    } for d, delta, passed in found[:MAX_OPTIONS]]

    if not options:
        return result(f"В пределах ±{WINDOW_DAYS} дней от {fmt_date(req.date)} нет даты, "
                      f"когда подходит больше подрядчиков.")
    listed = "; ".join(f"{o['label']} ({o['weekday']}) — {_variants(min(o['passed'], MAX_CARDS))}"
                       for o in options)
    n = len(base_passed)
    now = (f"подходит {n} {_plural(n, 'подрядчик', 'подрядчика', 'подрядчиков')}" if n
           else "никто не подходит")
    return result(f"На {fmt_date(req.date)} {now}. Ближайшие даты с большей подборкой: {listed}.", options)
