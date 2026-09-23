"""Офлайн-косинусная релевантность по заранее сохранённым векторам."""
from __future__ import annotations

import json
import math
from pathlib import Path

EMBEDDINGS_FILE = Path(__file__).resolve().parent.parent / "data" / "embeddings.json"
MODEL = "text-embedding-3-small"
FORMAT_PHRASES = {
    "свадьба": "свадьба, свадебное торжество, молодожёны, жених и невеста, церемония бракосочетания",
    "той": "той, казахский семейный праздник, беташар, ұзату, национальные традиции и обряды",
    "корпоратив": "корпоратив, праздник компании для сотрудников, тимбилдинг, бренд и бизнес-мероприятие",
    "конференция": "конференция, деловой форум, доклады спикеров, презентация и бизнес-встреча",
    "юбилей": "юбилей, памятная годовщина, торжество в честь юбиляра и гостей",
    "день рождения": "день рождения, праздник именинника, вечеринка и детский праздник",
}


def _cosine(left: list[float], right: list[float], right_norm: float) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def load_index(path: Path = EMBEDDINGS_FILE) -> dict | None:
    """Один раз готовит таблицу (формат, профиль) -> предложения по косинусу.

    При отсутствии файла оставляет старый лексический режим. При повреждённом
    файле не маскирует ошибку: это артефакт сборки, а не вход пользователя.
    """
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("model") != MODEL or set(payload.get("formats", {})) != set(FORMAT_PHRASES):
        raise ValueError("data/embeddings.json не соответствует модели или форматам")
    formats = payload["formats"]
    dimensions = len(next(iter(formats.values())))
    if not dimensions or any(len(vector) != dimensions for vector in formats.values()):
        raise ValueError("У форматов разные размеры эмбеддингов")
    result: dict[tuple[str, str], list[tuple[str, float]]] = {}
    for event_type, format_vector in formats.items():
        format_norm = math.sqrt(sum(value * value for value in format_vector))
        for profile_id, entries in payload["profiles"].items():
            scored = []
            for entry in entries:
                vector = entry["embedding"]
                if len(vector) != dimensions:
                    raise ValueError(f"Неверная размерность: {profile_id}")
                scored.append((entry["sentence"], _cosine(vector, format_vector, format_norm)))
            result[(profile_id, event_type)] = sorted(scored, key=lambda item: -item[1])
    return result


def ranked_sentences(index: dict | None, profile_id: str, description: str,
                     event_type: str) -> list[tuple[str, float]]:
    if index is None:
        return []
    # Не переносим вектор исходного профиля на изменённое описание в тестах/данных.
    ranked = index.get((profile_id, event_type), [])
    return [(text, score) for text, score in ranked if text in description]


# Считаем попарные косинусы при старте процесса, а не при каждом /api/match.
INDEX = load_index()
