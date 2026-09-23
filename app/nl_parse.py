"""Разбор запроса свободным текстом в параметры подбора.

Правила по словарю каталога: работают без сети и детерминированно. Для каждого поля
сохраняется фрагмент текста, из которого оно взято, — пользователь видит, что понял
сервис, и дозаполняет недостающее в форме.
"""
from __future__ import annotations

import re
from datetime import date

from .data import CALENDAR_END, CALENDAR_START
from .matcher import fmt_date

FLAGS = re.IGNORECASE | re.UNICODE
REQUIRED = ("city", "date", "event_type", "category", "budget")

MONTHS = [("январ", 1), ("феврал", 2), ("март", 3), ("апрел", 4), ("ма[йя]", 5), ("июн", 6),
          ("июл", 7), ("август", 8), ("сентябр", 9), ("октябр", 10), ("ноябр", 11), ("декабр", 12)]

CITIES = {
    "Алматы": [r"алмат", r"алма-ат"],
    "Астана": [r"астан", r"нур-султан", r"нурсултан"],
    "Зарубежье": [r"зарубеж", r"за границ", r"за рубеж"],
}

EVENT_TYPES = {
    "свадьба": [r"свадьб", r"свадеб", r"венчани", r"никах"],
    "той": [r"\bто[йяеюи]\b", r"\bтоев\b", r"ұзату", r"узату", r"беташар", r"с[үу]ндет", r"т[ұу]сау"],
    "корпоратив": [r"корпоратив", r"тимбилдинг", r"для (?:компании|сотрудников|коллег)"],
    "конференция": [r"конференц", r"форум", r"саммит", r"семинар"],
    "юбилей": [r"юбиле"],
    "день рождения": [r"дн[яеь]\s+рождени", r"день\s+рождени", r"\bдр\b", r"днюх"],
}

# Порядок важен: при совпадении в одной позиции побеждает более конкретная категория.
CATEGORIES = {
    "Ведущий церемонии": [r"ведущ\w*\s+церемони", r"церемониймейстер", r"выездн\w*\s+регистрац"],
    "Ведущий": [r"ведущ", r"тамад", r"конферансье", r"шоумен", r"модератор"],
    "Фото и видеобудки": [r"фото\s*-?\s*и\s+видеобудк", r"фото\s*-?\s*будк", r"видеобудк", r"\b360\b"],
    "Фотограф": [r"фотограф", r"фотосъ[её]мк", r"фотосесси", r"\bфото\b"],
    "Видеограф": [r"видеограф", r"видеосъ[её]мк", r"видеооператор", r"\bвидео\b"],
    "Банкетный зал": [r"банкетн", r"\bзал(?:а|у|е|ы|ов)?\b"],
    "Ресторан": [r"ресторан", r"\bкафе\b"],
    "Загородная площадка": [r"загородн", r"за город", r"на природе", r"коттедж", r"баз[аеуы] отдыха"],
    "Отель": [r"\bотел", r"гостиниц"],  # \b: иначе «хотел бы» = отель
    "Флорист": [r"флорист", r"цвет(?:ы|ов|ами|очн)", r"букет"],
    "Декоратор": [r"декор", r"оформлени"],
    "Подарки и сувениры": [r"подар", r"сувенир", r"бонбоньер"],
    "Танцевальный коллектив": [r"шоу-?\s*балет", r"танц", r"балет"],
    "Национальный ансамбль": [r"ансамбл", r"домбр", r"\bэтно", r"фольклор"],
    "Инструменталист": [r"инструменталист", r"саксофон", r"скрипа?ч", r"скрипк", r"пианист", r"арфист"],
    "Лайв-бэнд": [r"лайв", r"\blive\b", r"кавер", r"жив(?:ая|ую|ой)\s+музык", r"\bбэнд", r"\bband\b"],
    "Шоу-программа": [r"шоу", r"фаер", r"иллюзионист", r"фокусник"],
}

LANGUAGES = {
    "казахский": [r"на\s+казахском", r"казахоязычн", r"по-казахски", r"на\s+қазақ"],
    "английский": [r"на\s+английском", r"англоязычн", r"по-английски", r"\bin english\b"],
    "русский": [r"на\s+русском", r"русскоязычн", r"по-русски"],
}

NUM = r"\d+(?:[.,]\d+)?"
MONEY_UNIT = [
    (rf"({NUM})\s*(?:млн|миллион\w*|mln)(?!\w)", 1_000_000),
    (rf"({NUM})\s*(?:тыс\w*|тысяч\w*|к|k)(?!\w)", 1_000),
]
MONEY_PLAIN = r"(?<![\d.,])(\d{1,3}(?:[  ]\d{3})+|\d{5,})(?![\d.,])"


def _to_float(s: str) -> float:
    return float(s.replace(",", "."))


def _blank(text: str, span: tuple[int, int]) -> str:
    """Затирает найденный фрагмент, чтобы его цифры не разобрало следующее правило."""
    a, b = span
    return text[:a] + " " * (b - a) + text[b:]


NEGATION = re.compile(r"(?:^|(?<=\W))(не|без|кроме)\s+$", FLAGS)


def _negation(text: str, start: int) -> str | None:
    """«не на английском», «без фотографа», «кроме свадеб» — значение исключено, а не выбрано.

    Возвращает слово-отрицание перед позицией start или None.
    """
    m = NEGATION.search(text[max(0, start - 12):start])
    return m.group(1) if m else None


def _first_by_position(text: str, table: dict[str, list[str]],
                       negated: list[str] | None = None) -> list[tuple[int, str, str]]:
    """Совпадения словаря (позиция, значение, фрагмент): раньше в тексте — первым.

    Если два значения найдены на одном и том же месте («ведущий церемонии» и «ведущий»),
    остаётся более конкретное — то, что выше в словаре. Совпадения после «не/без/кроме»
    пропускаются и складываются в negated — фильтра «кроме» в сервисе нет.
    """
    hits = []
    for rank, (value, patterns) in enumerate(table.items()):
        best = None
        for p in patterns:
            for m in re.finditer(p, text, FLAGS):
                neg = _negation(text, m.start())
                if neg:
                    if negated is not None:
                        word_end = m.end() + re.match(r"\w*", text[m.end():]).end()
                        negated.append(f"{neg} {text[m.start():word_end]}")
                    continue
                if best is None or m.start() < best.start():
                    best = m
                break
        if best:
            # Паттерны — корни слов («астан», «свадьб»); пользователю показываем слово целиком.
            end = best.end() + re.match(r"\w*", text[best.end():]).end()
            hits.append((best.start(), rank, end, value, text[best.start():end]))
    hits.sort()
    taken: list[tuple[int, int]] = []
    out = []
    for start, _, end, value, frag in hits:
        if any(start < b and a < end for a, b in taken):
            continue
        taken.append((start, end))
        out.append((start, value, frag))
    return out


def _parse_date(text: str) -> tuple[date | None, str | None, tuple[int, int] | None, str | None]:
    m = re.search(r"\b(20\d\d)-(\d\d)-(\d\d)\b", text)
    if m:
        y, mo, d = int(m[1]), int(m[2]), int(m[3])
    else:
        months = "|".join(p for p, _ in MONTHS)
        m = re.search(rf"\b(\d{{1,2}})\s+({months})\w*(?:\s+(20\d\d))?", text, FLAGS)
        if m:
            d = int(m[1])
            mo = next(n for p, n in MONTHS if re.match(p, m[2], FLAGS))
            y = int(m[3]) if m[3] else CALENDAR_START.year
        else:
            m = re.search(r"\b(\d{1,2})[./](\d{1,2})(?:[./](\d{4}|\d{2}))?\b(?!\s*(?:млн|милл|тыс|к\b|k\b))",
                          text, FLAGS)
            if not m:
                return None, None, None, None
            d, mo = int(m[1]), int(m[2])
            y = int(m[3]) + (2000 if m[3] and len(m[3]) == 2 else 0) if m[3] else CALENDAR_START.year
    try:
        dt = date(y, mo, d)
    except ValueError:
        return None, None, None, f"«{m.group(0)}» не похоже на существующую дату"
    note = None
    if not (CALENDAR_START <= dt <= CALENDAR_END):
        note = (f"Дата {fmt_date(dt)} {dt.year} вне календаря подрядчиков "
                f"({fmt_date(CALENDAR_START)} — {fmt_date(CALENDAR_END)} {CALENDAR_END.year}).")
    return dt, m.group(0), m.span(), note


def _parse_budget(text: str) -> tuple[int | None, str | None, str | None]:
    """Бюджет, его фрагмент и замечание.

    «-800 тысяч» — отрицательная сумма: не принимается. «600-800 тысяч» — диапазон:
    берётся верхняя граница, потому что фильтр сравнивает «цену от» с максимумом бюджета.
    """
    found = []
    for pattern, mult in MONEY_UNIT:
        for m in re.finditer(pattern, text, FLAGS):
            found.append((m.start(), round(_to_float(m[1]) * mult), m.group(0)))
    if not found:
        for m in re.finditer(MONEY_PLAIN, text):
            found.append((m.start(), int(re.sub(r"\D", "", m[1])), m.group(0)))
    if not found:
        return None, None, None
    start, value, frag = min(found)
    before = text[:start]
    if re.search(r"\d\s*[-−–]\s*$", before):
        low = re.search(r"(\d+(?:[.,]\d+)?)\s*[-−–]\s*$", before)
        return value, f"{low.group(0)}{frag}", "Указан диапазон бюджета — взята верхняя граница."
    if re.search(r"[-−–]\s*$", before):
        return None, None, f"Бюджет «-{frag}» отрицательный — укажите сумму в тенге."
    if value <= 0:
        return None, None, "Бюджет должен быть больше нуля."
    return value, frag, None


def _parse_duration(text: str) -> tuple[int | None, str | None]:
    for p in (r"(?<!\d)(\d{1,2})\s*(?:-\s*)?(?:час\w*|ч\b)", r"час\w*\s+на\s+(\d{1,2})\b", r"на\s+(\d{1,2})\s+час"):
        m = re.search(p, text, FLAGS)
        if m and 1 <= int(m[1]) <= 24:
            return int(m[1]), m.group(0)
    return None, None


def parse_request(text: str, known: dict) -> dict:
    """known — ответ /api/meta: допустимые города, категории, форматы и языки."""
    params: dict = {k: None for k in (*REQUIRED, "duration", "language")}
    found: list[dict] = []
    notes: list[str] = []
    rest = text

    def put(field: str, value, fragment: str) -> None:
        params[field] = value
        found.append({"field": field, "value": value, "fragment": fragment.strip()})

    dt, frag, span, note = _parse_date(rest)
    if dt:
        put("date", dt.isoformat(), frag)
        rest = _blank(rest, span)
    if note:
        notes.append(note)

    duration, frag = _parse_duration(rest)
    if duration:
        put("duration", duration, frag)
        rest = rest.replace(frag, " " * len(frag), 1)

    budget, frag, note = _parse_budget(rest)
    if budget:
        put("budget", budget, frag)
    if note:
        notes.append(note)

    for field, table, allowed in (("city", CITIES, known["cities"]),
                                  ("event_type", EVENT_TYPES, known["event_types"]),
                                  ("category", CATEGORIES, known["categories"]),
                                  ("language", LANGUAGES, known["languages"])):
        negated: list[str] = []
        hits = [h for h in _first_by_position(text, table, negated) if h[1] in allowed]
        if negated:
            notes.append(f"«{negated[0]}» — исключить значение нельзя, сервис подбирает только "
                         f"по выбранному; это условие не учтено.")
        if not hits:
            continue
        put(field, hits[0][1], hits[0][2])
        if len(hits) > 1:
            extra = ", ".join(f"«{h[1]}»" for h in hits[1:])
            notes.append(f"В тексте есть и {extra} — сервис подбирает по одному значению "
                         f"(взято «{hits[0][1]}»), остальное запросите отдельно.")

    # «Казахская свадьба» не говорит ни о языке ведения, ни о формате «той» однозначно —
    # не угадываем, а подсказываем обе возможности.
    kz = re.search(r"казахск\w*\s+(?:свадьб|свадеб)\w*", text, FLAGS)
    if kz:
        tips = []
        if params["language"] is None:
            tips.append("если нужен подрядчик на казахском языке, выберите язык «казахский»")
        if params["event_type"] == "свадьба":
            tips.append("для традиционного тоя в каталоге есть отдельный формат «той»")
        if tips:
            notes.append(f"«{kz.group(0)}»: " + "; ".join(tips) + ".")

    labels = {"city": "город", "date": "дата", "event_type": "тип мероприятия",
              "category": "категория подрядчика", "budget": "бюджет"}
    missing = [f for f in REQUIRED if params[f] is None]
    if missing:
        notes.append("Не удалось понять: " + ", ".join(labels[f] for f in missing) + " — уточните в форме.")
    return {"params": params, "found": found, "missing": missing, "notes": notes, "source": "rules"}
