"""Воспроизводимые HTTP-сценарии: запустите сервер, затем python scripts/demo.py."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE = dict(city="Алматы", date="2026-10-10", event_type="свадьба",
            category="Фотограф", budget=800000, duration=None, language=None)
CASES = [
    ("dense", "Плотная категория: фотографы", {}, "found", ["HK-30583", "HK-16628", "HK-53108"]),
    ("rare", "Редкая категория: флористы", {"category": "Флорист"}, "partial", ["HK-39372"]),
    ("no-category", "Лайв-бэнд в Астане", {"city": "Астана", "category": "Лайв-бэнд"}, "no_category_in_city", []),
    ("none", "Бюджет исключает всех", {"budget": 1}, "none_match", []),
    ("december", "Тот же запрос на декабрь", {"date": "2026-12-12"}, "partial", ["HK-76268"]),
    ("invalid", "Дата вне календаря", {"date": "2027-01-01"}, "invalid_request", []),
]


def word_form(count: int, one: str, few: str, many: str) -> str:
    if 11 <= count % 100 <= 14:
        return many
    return {1: one, 2: few, 3: few, 4: few}.get(count % 10, many)


def post(base_url: str, body: dict) -> dict:
    req = Request(base_url.rstrip("/") + "/api/match",
                  data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                  headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=45) as response:
        return json.load(response)


def validate(data: dict, outcome: str, ids: list[str]) -> list[str]:
    errors = []
    if data.get("outcome") != outcome:
        errors.append(f"outcome: ожидался {outcome}, получен {data.get('outcome')}")
    cards = data.get("cards", [])
    actual_ids = [c.get("id") for c in cards]
    if actual_ids != ids:
        errors.append(f"карточки: ожидались {ids}, получены {actual_ids}")
    if not data.get("message"):
        errors.append("пустое message")
    if any(not c.get("explanation") for c in cards):
        errors.append("карточка без объяснения")
    return errors


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--case", choices=[x[0] for x in CASES], help="Запустить один сценарий")
    parser.add_argument("--save", type=Path, help="Сохранить фактические запросы и ответы в JSON")
    args = parser.parse_args()
    results, failures = {}, []
    for key, title, overrides, outcome, ids in CASES:
        if args.case and key != args.case:
            continue
        body = {**BASE, **overrides}
        print(f"\n=== {key}: {title} ===")
        print(json.dumps(body, ensure_ascii=False))
        try:
            data = post(args.base_url, body)
        except HTTPError as exc:
            print(f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}")
            failures.append(key)
            continue
        except (URLError, TimeoutError, OSError, ValueError) as exc:
            print(f"API недоступен или ответ некорректен: {exc}")
            failures.append(key)
            continue
        results[key] = {"request": body, "response": data}
        print(f"outcome: {data.get('outcome')}\nmessage: {data.get('message')}")
        funnel = data.get("funnel")
        if funnel:
            print("Воронка: " + " → ".join(
                f"{step['count']} {step['step']}" for step in funnel
            ))
        for c in data.get("cards", []):
            badge = " [синтетический профиль]" if c.get("synthetic") else ""
            price = f"{c['price_from_kzt']:,}".replace(",", " ")
            print(f"  {c['id']} · {c['name']}{badge} · от {price} ₸")
            print(f"  Почему: {c.get('explanation', '')}")
            if c.get("explanation_source"):
                print(f"  Источник объяснения: {c['explanation_source']}")
        for e in data.get("excluded", []):
            print(f"  Не попал {e['id']} · {e['name']}: {'; '.join(e['reasons'])}")
        for hint in data.get("hints", []):
            print(f"  Подсказка: {hint}")
        errors = validate(data, outcome, ids)
        if key == "none" and not data.get("excluded"):
            errors.append("ожидались причины исключения")
        failures.extend(f"{key}: {error}" for error in errors)
        print("FAIL: " + "; ".join(errors) if errors else "PASS")
    if "dense" in results and "december" in results:
        october = {c["id"] for c in results["dense"]["response"]["cards"]}
        busy = {e["id"] for e in results["december"]["response"]["excluded"]
                if any("занят" in reason.lower() for reason in e["reasons"])}
        disappeared = sorted(october & busy)
        print(f"\nОктябрь → декабрь: заняты {', '.join(disappeared) or 'никто'}")
        if not october or not october.issubset(busy):
            failures.append("Смена даты: ожидалось исключение всех трёх октябрьских кандидатов по занятости")
    if args.save and not failures:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    elif args.save:
        print("Результаты не сохранены: проверка сценариев завершилась с ошибками.")
    print(f"\n{'FAIL' if failures else 'PASS'}: {len(results)} "
          f"{word_form(len(results), 'ответ', 'ответа', 'ответов')}, "
          f"{len(failures)} {word_form(len(failures), 'ошибка', 'ошибки', 'ошибок')}")
    for failure in failures:
        print(f"  {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
