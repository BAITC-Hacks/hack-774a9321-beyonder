"""Пять демонстрационных запросов с настоящим LLM-ключом и свежим временным кэшем.

Запуск из корня проекта: python scripts/evaluate_explanations.py
Ключ читается только из окружения или локального .env и никогда не печатается.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

from dotenv import load_dotenv
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env", override=False)

from app import explain_llm  # noqa: E402
from app.main import app  # noqa: E402
from scripts.demo import BASE, CASES, validate  # noqa: E402


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    key_name = {"openai": "OPENAI_API_KEY", "nvidia": "NVIDIA_API_KEY"}.get(provider)
    if not key_name or not os.getenv(key_name, "").strip():
        print(f"Нет ключа для провайдера {provider!r}: добавьте его в {ROOT / '.env'}")
        return 2
    if provider == "nvidia" and not os.getenv("NVIDIA_MODEL", "").strip():
        print("Для NVIDIA также нужен NVIDIA_MODEL в .env")
        return 2
    os.environ["EXPLANATIONS_LLM_ENABLED"] = "true"
    errors = []
    with TemporaryDirectory(prefix="beyonder-live-eval-") as directory:
        explain_llm.CACHE_FILE = Path(directory) / "explanations.json"
        explain_llm._MEMORY.clear()
        with TestClient(app) as client:
            for name, _, overrides, outcome, ids in CASES[:5]:
                body = {**BASE, **overrides}
                start = perf_counter()
                response = client.post("/api/match", json=body)
                elapsed = perf_counter() - start
                if response.status_code != 200:
                    errors.append(f"{name}: HTTP {response.status_code}")
                    print(f"{name}: HTTP {response.status_code}, {elapsed:.2f} с")
                    continue
                data = response.json()
                errors.extend(f"{name}: {error}" for error in validate(data, outcome, ids))
                if elapsed >= 10:
                    errors.append(f"{name}: {elapsed:.2f} с, больше 10 с")
                print(f"{name}: {data['outcome']}, {elapsed:.2f} с")
                for card in data["cards"]:
                    explanation = card["explanation"]
                    source = card.get("explanation_source")
                    print(f"  {card['id']} [{source}, {len(explanation)} знаков] {explanation}")
                    if source != "llm":
                        errors.append(f"{name}/{card['id']}: LLM недоступен или ответ отклонён")
                    if not isinstance(card.get("highlights"), list) or not card["highlights"]:
                        errors.append(f"{name}/{card['id']}: отсутствуют highlights")
                    if len(explanation) > 220 or not 1 <= explain_llm._sentence_count(explanation) <= 2:
                        errors.append(f"{name}/{card['id']}: нарушена длина объяснения")
                if len({card["explanation"] for card in data["cards"]}) != len(data["cards"]):
                    errors.append(f"{name}: одинаковые объяснения")
    print(f"{'PASS' if not errors else 'FAIL'}: 5 запросов, {len(errors)} ошибок")
    for error in errors:
        print("  " + error)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
