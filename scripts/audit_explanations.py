"""Аудит объяснений по требованиям задания на всей сетке запросов.

Перебирает все пары «город × категория» из каталога, все форматы, даты через каждые
7 дней календаря и три уровня бюджета. Для каждой выдачи проверяет:
  - карточки одного запроса не повторяют друг друга (ни целиком, ни первым предложением);
  - нет общих фраз из стоп-листа;
  - не больше двух предложений и не длиннее 300 символов;
  - время ответа.

По умолчанию работает в процессе, без сети и с выключенным LLM (шаблонные объяснения).
Запуск из корня репозитория:
    python scripts/audit_explanations.py            # сводка
    python scripts/audit_explanations.py --examples 5   # плюс примеры нарушений
    python scripts/audit_explanations.py --strict   # код 1, если есть нарушения
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

STOP_PHRASES = ["отличный выбор", "идеально подойд", "профессионал своего дела", "лучший выбор",
                "станет украшением", "не пожалеете", "подойдёт для вашего мероприятия",
                "подойдет для вашего мероприятия", "высокий уровень сервиса"]
MAX_CHARS = 300
BUDGETS = (500_000, 1_500_000, 5_000_000)


def sentences(text: str) -> list[str]:
    # Точка внутри «цитаты» не завершает предложение.
    outside = re.sub(r"«[^»]*»", "«…»", text)
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[А-ЯЁA-Z«])", outside.strip()) if s]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--examples", type=int, default=3, help="сколько примеров показать на каждое нарушение")
    ap.add_argument("--step-days", type=int, default=7, help="шаг перебора дат")
    ap.add_argument("--strict", action="store_true", help="код выхода 1, если найдены нарушения")
    args = ap.parse_args()

    os.environ.setdefault("EXPLANATIONS_LLM_ENABLED", "false")
    from app.data import CALENDAR_END, CALENDAR_START, load_catalog
    from app.matcher import MatchRequest, match

    catalog = load_catalog()
    pairs = sorted({(c.city, cat) for c in catalog for cat in c.categories})
    formats = sorted({f for c in catalog for f in c.formats})
    dates = []
    d = CALENDAR_START
    while d <= CALENDAR_END:
        dates.append(d)
        d += timedelta(days=args.step_days)

    outcomes: Counter = Counter()
    problems: dict[str, list[str]] = defaultdict(list)
    cards_total, times = 0, []

    for city, category in pairs:
        for fmt in formats:
            for day in dates:
                for budget in BUDGETS:
                    req = MatchRequest(city=city, date=day, event_type=fmt, category=category, budget=budget)
                    t = time.perf_counter()
                    r = match(req, catalog)
                    times.append(time.perf_counter() - t)
                    outcomes[r["outcome"]] += 1
                    cards = r["cards"]
                    cards_total += len(cards)
                    where = f"{city}/{category}/{fmt}/{day:%d.%m}/{budget:,}".replace(",", " ")
                    texts = [c["explanation"] for c in cards]
                    if len(set(texts)) < len(texts):
                        problems["одинаковые объяснения в одной выдаче"].append(f"{where}: {texts[0]}")
                    firsts = [sentences(t)[0] if sentences(t) else t for t in texts]
                    if len(cards) > 1 and len(set(firsts)) < len(firsts):
                        dup = Counter(firsts).most_common(1)[0][0]
                        problems["одинаковое первое предложение"].append(f"{where}: {dup}")
                    for c in cards:
                        text, low = c["explanation"], c["explanation"].lower()
                        hit = next((p for p in STOP_PHRASES if p in low), None)
                        if hit:
                            problems["общая фраза из стоп-листа"].append(f"{where} · {c['name']}: «{hit}»")
                        if len(sentences(text)) > 2:
                            problems["больше двух предложений"].append(f"{where} · {c['name']}: {text}")
                        if len(text) > MAX_CHARS:
                            problems[f"длиннее {MAX_CHARS} символов"].append(f"{where} · {c['name']}: {len(text)}")

    total = sum(outcomes.values())
    times.sort()
    print(f"Запросов: {total} · карточек: {cards_total} · "
          f"время: медиана {times[len(times) // 2] * 1000:.2f} мс, максимум {times[-1] * 1000:.1f} мс")
    print("Исходы: " + ", ".join(f"{k} {v}" for k, v in outcomes.most_common()))
    if not problems:
        print("Нарушений не найдено.")
        return 0
    print("Нарушения:")
    for name, items in sorted(problems.items(), key=lambda kv: -len(kv[1])):
        print(f"  - {name}: {len(items)}")
        for ex in items[:args.examples]:
            print(f"      {ex}")
    return 1 if args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
