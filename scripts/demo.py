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
    ("dense", "Плотная категория: фотографы", {}, "found", ["HK-30583", "HK-53108", "HK-16628"]),
    ("rare", "Редкая категория: флористы", {"category": "Флорист"}, "partial", ["HK-39372"]),
    ("no-category", "Лайв-бэнд в Астане", {"city": "Астана", "category": "Лайв-бэнд"}, "no_category_in_city", []),
    ("none", "Бюджет исключает всех", {"budget": 1}, "none_match", []),
    ("december", "Тот же запрос на декабрь", {"date": "2026-12-12"}, "partial", ["HK-76268"]),
    ("invalid", "Дата вне календаря", {"date": "2027-01-01"}, "invalid_request", []),
]
NL_TEXT = "Нужен ведущий на казахскую свадьбу 14 ноября в Алматы, бюджет до 800 тысяч, часов на 6"
NL_EXPECTED = dict(city="Алматы", date="2026-11-14", event_type="свадьба",
                   category="Ведущий", budget=800000, duration=6, language=None)
PLAN_B = {**BASE, "date": "2026-12-12"}


def word_form(count: int, one: str, few: str, many: str) -> str:
    if 11 <= count % 100 <= 14:
        return many
    return {1: one, 2: few, 3: few, 4: few}.get(count % 10, many)


def fmt_kzt(amount: int) -> str:
    return f"{amount:,}".replace(",", " ") + " ₸"


def post(base_url: str, body: dict, path: str = "/api/match") -> dict:
    req = Request(base_url.rstrip("/") + path,
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


def show_match(data: dict) -> None:
    print(f"outcome: {data.get('outcome')}\nmessage: {data.get('message')}")
    funnel = data.get("funnel")
    if funnel:
        print("Воронка: " + " → ".join(
            f"{step['count']} {step['step']}" for step in funnel
        ))
    for card in data.get("cards", []):
        badge = " [синтетический профиль]" if card.get("synthetic") else ""
        print(f"  {card['id']} · {card['name']}{badge} · от {fmt_kzt(card['price_from_kzt'])}")
        print(f"  Почему: {card.get('explanation', '')}")
        if card.get("explanation_source"):
            print(f"  Источник объяснения: {card['explanation_source']}")
    for excluded in data.get("excluded", []):
        print(f"  Не попал {excluded['id']} · {excluded['name']}: {'; '.join(excluded['reasons'])}")
    for hint in data.get("hints", []):
        print(f"  Подсказка: {hint}")


def check_nl(base_url: str, results: dict, failures: list[str]) -> None:
    """Свободный текст → /api/ask: поля распознаны, неоднозначность показана, подбор выполнен."""
    print(f"\n=== nl: Запрос свободным текстом ===\n{NL_TEXT}")
    try:
        data = post(base_url, {"text": NL_TEXT}, "/api/ask")
    except HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}")
        failures.append("nl")
        return
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        print(f"API недоступен или ответ некорректен: {exc}")
        failures.append("nl")
        return
    results["nl"] = {"request": {"text": NL_TEXT}, "response": data}
    parsed, result = data.get("parsed") or {}, data.get("result") or {}
    for item in parsed.get("found", []):
        print(f"  понял: {item['field']} = {item['value']}  ← «{item['fragment']}»")
    for note in parsed.get("notes", []):
        print(f"  Уточнение: {note}")
    if parsed.get("missing"):
        print(f"  Не хватает: {', '.join(parsed['missing'])}")
    show_match(result)
    errors = [f"{key}: ожидалось {value}, получено {parsed.get('params', {}).get(key)}"
              for key, value in NL_EXPECTED.items() if parsed.get("params", {}).get(key) != value]
    if parsed.get("missing"):
        errors.append(f"не распознаны: {', '.join(parsed['missing'])}")
    if parsed.get("source") != "rules":
        errors.append(f"источник разбора: ожидался rules, получен {parsed.get('source')}")
    if not any("казахский" in note for note in parsed.get("notes", [])):
        errors.append("нет подсказки о языке для «казахской свадьбы»")
    errors.extend(validate(result, "partial", ["HK-44923"]))
    failures.extend(f"nl: {error}" for error in errors)
    print("FAIL: " + "; ".join(errors) if errors else "PASS")


def check_alternatives(base_url: str, results: dict, failures: list[str]) -> None:
    """«План Б» → /api/alternatives: на занятую дату предлагаются ближайшие даты с полной подборкой."""
    print(f"\n=== plan-b: «План Б» для 12 декабря ===\n{json.dumps(PLAN_B, ensure_ascii=False)}")
    try:
        data = post(base_url, PLAN_B, "/api/alternatives")
    except HTTPError as exc:
        if exc.code == 404:
            print("SKIP: на сервере нет /api/alternatives")
            return
        print(f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}")
        failures.append("plan-b")
        return
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        print(f"API недоступен или ответ некорректен: {exc}")
        failures.append("plan-b")
        return
    results["plan-b"] = {"request": PLAN_B, "response": data}
    print(f"message: {data.get('message')}")
    options = data.get("options") or []
    for option in options:
        print(f"  {option['label']} ({option['weekday']}): проходят {option['passed']} — {', '.join(option['top'])}")
    errors = []
    if (data.get("base") or {}).get("passed") != 1:
        errors.append(f"на 12 декабря ожидался 1 подходящий, получено {(data.get('base') or {}).get('passed')}")
    if not options or options[0].get("date") != "2026-12-13" or options[0].get("passed", 0) < 3:
        errors.append("первой ожидалась дата 2026-12-13 с полной подборкой")
    if not data.get("message"):
        errors.append("пустое message")
    failures.extend(f"plan-b: {error}" for error in errors)
    print("FAIL: " + "; ".join(errors) if errors else "PASS")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--case", choices=[x[0] for x in CASES] + ["nl", "plan-b"], help="Запустить один сценарий")
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
        show_match(data)
        errors = validate(data, outcome, ids)
        if key == "none" and not data.get("excluded"):
            errors.append("ожидались причины исключения")
        failures.extend(f"{key}: {error}" for error in errors)
        print("FAIL: " + "; ".join(errors) if errors else "PASS")
    if "dense" in results and "december" in results:
        october = {card["id"] for card in results["dense"]["response"]["cards"]}
        busy = {excluded["id"] for excluded in results["december"]["response"]["excluded"]
                if any("занят" in reason.lower() for reason in excluded["reasons"])}
        disappeared = sorted(october & busy)
        print(f"\nОктябрь → декабрь: заняты {', '.join(disappeared) or 'никто'}")
        if not october or not october.issubset(busy):
            failures.append("Смена даты: ожидалось исключение всех трёх октябрьских кандидатов по занятости")
    if not args.case or args.case == "nl":
        check_nl(args.base_url, results, failures)
    if not args.case or args.case == "plan-b":
        check_alternatives(args.base_url, results, failures)
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
