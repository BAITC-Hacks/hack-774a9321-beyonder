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
from . import explain_llm
from . import semantic

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

VENUE_CATEGORIES = {"Банкетный зал", "Ресторан", "Отель", "Загородная площадка"}
CONTEXT_DEPENDENT_LEAD = re.compile(
    r"^(?:такой|такая|такое|такие|этот|эта|это|эти|он|она|они|там|поэтому|"
    r"его|её|ее|их|который|которая|которое|которые)\b", re.I,
)

WEIGHTS = {"budget": 0.40, "relevance": 0.30, "hours": 0.15, "language": 0.10, "data_quality": 0.05}


def fmt_date(d: date) -> str:
    return f"{d.day} {MONTHS_GEN[d.month - 1]}"


def fmt_kzt(v: int) -> str:
    return f"{v:,}".replace(",", " ") + " ₸"


def _plural(n: int, one: str, few: str, many: str) -> str:
    if 11 <= n % 100 <= 14:
        return many
    return {1: one, 2: few, 3: few, 4: few}.get(n % 10, many)


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
    parts = re.split(r"(?<=[.!?])\s+|[•·;]+|\n+", text)
    # Для индекса нужны и короткие предложения; неполезные цитаты ниже
    # отсеиваются отдельно по длине и содержанию.
    return [p.strip() for p in parts if p.strip()]


def _truncate_quote(text: str, limit: int) -> str | None:
    """Не разрывает слово и явно помечает любое сокращение многоточием."""
    if len(text) <= limit:
        return text
    if limit < 2:
        return None
    prefix = text[:limit - 1]
    boundary = prefix.rfind(" ")
    if boundary < 0:
        return None
    whole_words = prefix[:boundary].rstrip(" ,;:.!?…")
    return whole_words + "…" if whole_words else None


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
    if semantic.INDEX is None:
        cand.snippet, cand.marker_hits = _best_snippet(c.description, req.event_type)
        specialist = 0.3 if len(c.formats) <= 2 else 0.0
        relevance = min(1.0, cand.marker_hits / 3) * 0.7 + specialist
        semantic_score = 0.0
    else:
        ranked = semantic.ranked_sentences(semantic.INDEX, c.id, c.description, req.event_type)
        cand.snippet = ranked[0][0] if ranked else None
        cand.marker_hits = 0
        semantic_score = ranked[0][1] if ranked else 0.0
        relevance = semantic_score
    if req.duration and c.max_hours is not None:
        hours = min(1.0, (c.max_hours - req.duration) / 4)
    elif c.max_hours is None:
        hours = 1.0
    else:
        hours = 0.5
    language = 1.0 if req.language else len(c.languages) / 3
    data_quality = 1.0 - 0.5 * c.price_imputed - 0.5 * c.city_imputed
    cand.parts = {"budget": budget, "relevance": relevance, "semantic": semantic_score,
                  "hours": hours, "language": language, "data_quality": data_quality}
    cand.score = round(sum(WEIGHTS[k] * cand.parts[k] for k in WEIGHTS), 6)


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
        return ["единственный подходящий профиль"]
    facts = []
    other_languages = set().union(*(o.c.languages for o in others))
    for language in sorted(c.languages - other_languages):
        facts.append(f"единственный в подборке работает {LANG_INSTR.get(language, language)}")
    low, high = min(o.c.price for o in others), max(o.c.price for o in others)
    if c.price < low:
        facts.append(f"на {fmt_kzt(low - c.price)} дешевле ближайшего")
    elif c.price > high:
        facts.append(f"на {fmt_kzt(c.price - high)} дороже ближайшего")
    elif low < c.price < high:
        facts.append(f"на {fmt_kzt(c.price - low)} дороже самого доступного")
    elif c.price == low < high:
        facts.append(f"на {fmt_kzt(high - c.price)} дешевле самого дорогого")
    elif low < c.price == high:
        facts.append(f"на {fmt_kzt(c.price - low)} дороже самого доступного")
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
    """Дословное свидетельство формата, опыта или масштаба длиной до 100 знаков."""
    others = [o.c.description.casefold() for o in shown if o is not cand]
    venue = req.category in VENUE_CATEGORIES
    semantic_scores = dict(semantic.ranked_sentences(
        semantic.INDEX, cand.c.id, cand.c.description, req.event_type))
    options = []
    for position, sentence in enumerate(_sentences(cand.c.description)):
        if CONTEXT_DEPENDENT_LEAD.match(sentence):
            continue
        if others and any(sentence.casefold() in desc for desc in others):
            continue
        semantic_score = semantic_scores.get(sentence)
        first_person = bool(re.search(
            r"\b(?:я|мы|мне|меня|нас|нам|наш\w*|мо[йяеи]|работаем|снимаем|"
            r"созда[её]м|организуем|предлагаем|прославляем)\b", sentence, re.I))
        letters = [ch for ch in sentence if ch.isalpha()]
        all_caps = bool(letters and sum(ch.isupper() for ch in letters) / len(letters) > 0.6)
        fragments = ([sentence] if len(sentence) <= 100 else
                     [part.strip() for part in re.split(
                         r"[,;:]|(?=\b(?:Финалист|Резидент|Ведущий|Организатор|"
                         r"Участник|Сценарист)\b)", sentence,
                     )])
        for fragment in fragments:
            fragment = fragment.strip()
            if CONTEXT_DEPENDENT_LEAD.match(fragment):
                continue
            fragment = re.sub(rf"^{re.escape(cand.c.name)}\s*[—–-]\s*", "", fragment, flags=re.I)
            named_lead = re.match(r"^([^—–-]{2,40})\s+[—–]\s+(.+)$", fragment)
            if named_lead and len(named_lead[1].split()) <= 4 and all(
                    word[0].isupper() for word in named_lead[1].split()):
                fragment = named_lead[2]
            if CONTEXT_DEPENDENT_LEAD.match(fragment):
                continue
            if len(fragment) > 100:
                fragment = _truncate_quote(fragment, 100)
                if fragment is None:
                    continue
            fragment = fragment.rstrip(",;:.!? ")
            if (len(fragment) < 16 or "«" in fragment or "»" in fragment
                    or explain_llm.has_generic_phrase(fragment)):
                continue
            event_hits = _marker_hits(fragment, req.event_type) if semantic.INDEX is None else 0
            capacity = bool(re.search(
                r"\b\d+\s*(?:гост(?:ей|я|ь)?|человек|мест)\b|"
                r"\b(?:вместимост[ьи]|зал\s+на)\b.{0,30}\b\d+", fragment, re.I,
            ))
            ranking = bool(re.search(r"\b(?:топ|top)\s*[-–]?\s*\d+", fragment, re.I))
            quantitative = ranking or bool(re.search(
                r"\b\d+\s*(?:лет|года?|свадеб|мероприяти\w*|заказ\w*|"
                r"проект\w*|съ[её]м\w*)\b", fragment, re.I,
            )) or bool(req.category in {"Лайв-бэнд", "Национальный ансамбль"}
                       and re.search(r"\b\d{2,4}-х\b", fragment))
            scale = bool(re.search(r"\b\d+\s*(?:лет|год|заказ|проект|событ)|\b(?:семи|пяти|десяти)\s+лет\b|\bопыт", fragment, re.I))
            subject = bool(re.search(
                r"фотограф|флорист|цветочн|оформлен|съ[её]мк|снима|вед[её]т|дизайн|"
                r"музык|заказ|команд|клиент|стаж|лет|ресторан|банкет|отел|площадк|"
                r"локац|террас|панорам|кухн|интерьер|гор[аые]|вилл|декор|подар|"
                r"сувенир|танц|ансамбл|шоу|сцен|звук|артист|видео|свет|гост|зал|"
                r"преми|финалист|резидент|сценарист|ведущ|форум|конференц|квн|"
                r"песн|репертуар|выступ|коллектив|концерт|перформанс|хит",
                fragment, re.I,
            ))
            if not (event_hits or scale or subject or quantitative or semantic_score is not None):
                continue
            unique = all(fragment.casefold() not in desc for desc in others)
            action = bool(re.search(r"работа|снима|вед[её]|явля|специализ|реализ|организ|оформл", fragment, re.I))
            music_detail = bool(req.category in {"Лайв-бэнд", "Национальный ансамбль"}
                                and re.search(r"репертуар|хит|песн|вокал|состав|выступ", fragment, re.I))
            venue_identity = bool(venue and re.search(
                r"\b(?:ресторан\w*|локаци\w*|площадк\w*|отел\w*|"
                r"курорт\w*|гольф|банкетн\w* зал)\b",
                fragment, re.I))
            original_rank = (bool(venue and capacity), quantitative,
                             bool(event_hits and action), scale, bool(event_hits), music_detail,
                             subject, -position, -len(fragment))
            # Косинус задаёт основу; содержательные доказательства (числа, репертуар,
            # тип и масштаб площадки) важнее общего «профессиональный коллектив».
            quote_score = (semantic_score or 0.0) + (0.30 if venue_identity else 0.0)
            quote_score += 0.30 if quantitative else 0.0
            quote_score += 0.26 if venue and capacity else 0.0
            quote_score += 0.22 if music_detail else 0.0
            quote_score += 0.04 if fragment[:1].isupper() else 0.0
            quote_score -= 0.08 if first_person else 0.0
            quote_score -= 0.07 if all_caps else 0.0
            ranking = ((unique, round(quote_score, 6), *original_rank)
                       if semantic.INDEX is not None else (unique, *original_rank))
            options.append((ranking, fragment))
    return max(options, default=((), None))[1]


def _explanation_facts(cand: Candidate, req: MatchRequest, shown: list[Candidate]) -> dict:
    c = cand.c
    comparisons = _comparisons(cand, shown)
    required_comparison = next((part for part in comparisons if "₸" in part), None)
    if required_comparison is None:
        required_comparison = next((part for part in comparisons if " ч" in part), None)
    return {
        "id": c.id, "price_from_kzt": c.price,
        "budget_percent": round(100 * c.price / req.budget),
        "price_imputed": c.price_imputed,
        "event_formats": sorted(c.formats), "languages": sorted(c.languages),
        "max_hours": c.max_hours, "requested_duration": req.duration,
        "description_quote": _description_quote(cand, req, shown),
        "comparisons": comparisons, "required_comparison": required_comparison,
    }


def _explain(cand: Candidate, req: MatchRequest, facts: dict,
             common_max_hours: int | None) -> str:
    c = cand.c
    comparison = facts["required_comparison"]
    price = f"Цена от {fmt_kzt(c.price)} — {facts['budget_percent']}% бюджета"
    if c.price_imputed:
        price += " (цена оценочная)"
    second = price + (f"; {comparison}" if comparison else "") + "."
    quote = facts["description_quote"]
    if quote:
        # Сохраняем дословность цитаты и целые слова даже при тесной карточке.
        limit = min(100, 220 - len("В описании — «». ") - len(second))
        quote = _truncate_quote(quote, limit)
    if quote:
        first = f"В описании — «{quote}»."
    elif req.duration and c.max_hours is not None and c.max_hours != common_max_hours:
        first = f"До {c.max_hours} ч на площадке при запросе на {req.duration} ч."
    elif c.max_hours is not None and c.max_hours != common_max_hours:
        first = f"До {c.max_hours} ч на площадке."
    elif any(part.startswith("единственный в подборке работает") for part in facts["comparisons"]):
        first = next(part.capitalize() + "." for part in facts["comparisons"]
                     if part.startswith("единственный в подборке работает"))
    elif len(c.formats) == 1:
        first = f"В профиле указан только формат «{req.event_type}»."
    else:
        first = f"В профиле указаны языки: {', '.join(sorted(c.languages))}."
    if len(first) + 1 + len(second) > 220:
        second = price + "."
    return f"{first} {second}"


def _highlights(comparisons: list[str]) -> list[str]:
    """Превратить проверенные сравнения в короткие различимые UI-метки."""
    items, formats = [], []
    for part in comparisons:
        price = re.fullmatch(r"на (.+ ₸) (дешевле|дороже) (.+)", part)
        hours = re.fullmatch(r"до (\d+) ч на площадке, у остальных — не более \d+ ч", part)
        language = re.fullmatch(r"единственный в подборке работает на (.+)", part)
        event_format = re.fullmatch(r"единственный в подборке также берёт формат «(.+)»", part)
        if price:
            items.append(f"{price[2].capitalize()} {price[3]} на {price[1]}")
        elif hours:
            items.append(f"До {hours[1]} ч — дольше остальных")
        elif language:
            instrumental = {"казахском": "казахским", "русском": "русским",
                            "английском": "английским"}.get(language[1], language[1])
            items.append(f"Единственный с {instrumental}")
        elif part == "единственный в подборке без привязки к часам присутствия на площадке":
            items.append("Только здесь лимит часов не указан")
        elif event_format:
            formats.append(f"Единственный также берёт «{event_format[1]}»")
        else:
            items.append(part.capitalize())
    return (items if items else formats)[:3]


def _card(cand: Candidate, explanation: str, highlights: list[str]) -> dict:
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
        "highlights": highlights,
    }


def _reason_summary(cands: list[Candidate], req: MatchRequest) -> str:
    counts: dict[str, int] = {}
    for cand in cands:
        for code, _ in cand.fails:
            if code == "busy":
                continue  # Занятость категории уже посчитана в общем сообщении.
            counts[code] = counts.get(code, 0) + 1
    labels = {
        "budget": f"дороже бюджета {fmt_kzt(req.budget)}",
        "format": lambda n: f"не {_plural(n, 'берёт', 'берут', 'берут')} формат «{req.event_type}»",
        "language": lambda n: f"не {_plural(n, 'работает', 'работают', 'работают')} {LANG_INSTR.get(req.language or '', req.language or '')}",
        "duration": lambda n: f"не {_plural(n, 'может', 'могут', 'могут')} работать {req.duration} ч",
    }
    parts = [f"{n} {labels[k](n) if callable(labels[k]) else labels[k]}"
             for k, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    return ", ".join(parts)


def _funnel(pool: list[Candidate], req: MatchRequest) -> list[dict]:
    """Последовательное сужение пула; причины отказа при этом остаются полными."""
    steps = [{"step": "в категории и городе", "count": len(pool)}]
    remaining = pool
    for code in FILTER_ORDER:
        if code == "language" and not req.language:
            continue
        if code == "duration" and req.duration is None:
            continue
        remaining = [p for p in remaining if code not in {k for k, _ in p.fails}]
        n = len(remaining)
        labels = {
            "busy": f"{_plural(n, 'свободен', 'свободны', 'свободны')} {fmt_date(req.date)}",
            "budget": "в бюджете",
            "format": f"{_plural(n, 'берёт', 'берут', 'берут')} формат «{req.event_type}»",
            "language": f"{_plural(n, 'работает', 'работают', 'работают')} {LANG_INSTR.get(req.language or '', req.language or '')}",
            "duration": f"{_plural(n, 'может', 'могут', 'могут')} работать {req.duration} ч",
        }
        steps.append({"step": labels[code], "count": n})
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
        hints.append(f"При бюджете от {fmt_kzt(cheapest.c.price)} подошёл бы профиль «{cheapest.c.name}»: свободен и берёт этот формат.")
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
    common_max_hours = (shown[0].c.max_hours if len(shown) > 1
                        and shown[0].c.max_hours is not None
                        and all(p.c.max_hours == shown[0].c.max_hours for p in shown)
                        else None)
    shared_quote = None
    if len(shown) > 1:
        quotes = [facts[p.c.id]["description_quote"] for p in shown]
        if quotes[0] and all(quote == quotes[0] for quote in quotes):
            shared_quote = quotes[0]
            for fact in facts.values():
                fact["description_quote"] = None
    highlights = [_highlights(facts[p.c.id]["comparisons"]) for p in shown]
    signatures = [tuple(items) for items in highlights]
    for index, p in enumerate(shown):
        if len(shown) > 1 and (not highlights[index] or signatures.count(signatures[index]) > 1):
            fallback = f"Позиция {index + 1} из {len(shown)} в подборке"
            facts[p.c.id]["comparisons"].append(fallback)
            highlights[index] = [*highlights[index][:2], fallback]
    cards = [_card(p, _explain(p, req, facts[p.c.id], common_max_hours),
                   highlights[index]) for index, p in enumerate(shown)]
    # Не выдумываем различия между полностью совпадающими профилями.
    texts = [card["explanation"] for card in cards]
    for card in cards:
        if texts.count(card["explanation"]) > 1:
            card["explanation"] = f"Профиль {card['id']}: " + card["explanation"]
    rewritten = explain_llm.rewrite(cards, req, {
        "facts_by_id": facts, "pool_size": len(pool), "busy_in_pool": busy_in_pool,
    })
    if rewritten is not None:
        for card, explanation in zip(cards, rewritten):
            card["explanation"] = explanation
            card["explanation_source"] = "llm"
    excluded = [{"id": f.c.id, "name": f.c.name, "synthetic": f.c.synthetic,
                 "reasons": [t for _, t in f.fails]} for f in sorted(failed, key=lambda x: x.c.id)]

    pool_desc = (f"Из {len(pool)} {_plural(len(pool), 'подрядчика', 'подрядчиков', 'подрядчиков')} "
                 f"категории «{req.category}» в городе {req.city}")
    occupancy = (f"в категории и городе {busy_in_pool} из {len(pool)} "
                 f"{_plural(busy_in_pool, 'занят', 'заняты', 'заняты')}.")
    shared = (f"На {fmt_date(req.date)} все показанные профили свободны и берут формат "
              f"«{req.event_type}»")
    if req.language:
        shared += f", работают {LANG_INSTR.get(req.language, req.language)}"
    shared += f"; {occupancy}"
    if len(shown) > 1:
        common_languages = set.intersection(*(set(p.c.languages) for p in shown))
        if len(common_languages) > 1 and not req.language:
            shared += (" У всех показанных профилей указаны языки: "
                       + ", ".join(sorted(common_languages)) + ".")
    if common_max_hours is not None:
        shared += (f" У всех показанных профилей максимум {common_max_hours} ч на площадке"
                   + (f" при запросе {req.duration} ч." if req.duration else "."))
    if shared_quote:
        shared += f" Во всех показанных описаниях — «{shared_quote}»."
    other_reasons = _reason_summary(failed, req)
    if not shown:
        outcome = "none_match"
        message = f"{pool_desc} ни один не проходит по условиям. На {fmt_date(req.date)} {occupancy}"
        if other_reasons:
            message += f" Другие причины отказа: {other_reasons}."
    elif len(shown) < MAX_CARDS:
        outcome = "partial"
        if failed:
            message = (f"{pool_desc} условиям {_plural(len(shown), 'соответствует', 'соответствуют', 'соответствуют')} "
                       f"только {len(shown)}. {shared}")
            if other_reasons:
                message += f" Другие причины отказа: {other_reasons}."
        else:
            message = (f"В городе {req.city} всего {len(pool)} "
                       f"{_plural(len(pool), 'подрядчик', 'подрядчика', 'подрядчиков')} категории «{req.category}» — "
                       f"все подходят, поэтому карточек меньше трёх. {shared}")
    else:
        outcome = "found"
        message = (f"{pool_desc} условиям {_plural(len(passed), 'соответствует', 'соответствуют', 'соответствуют')} "
                   f"{len(passed)}; показаны 3 лучших. {shared}")

    return {**base, "outcome": outcome, "message": message, "cards": cards,
            "excluded": excluded, "hints": _hints(failed, req) if len(shown) < MAX_CARDS else []}
