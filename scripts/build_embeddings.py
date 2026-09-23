"""Предрасчёт эмбеддингов описаний организаторов; рантайму ключ не нужен.

Запуск из корня репозитория: python scripts/build_embeddings.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.data import DATASET, load_catalog  # noqa: E402
from app.matcher import _sentences  # noqa: E402
from app.semantic import EMBEDDINGS_FILE, FORMAT_PHRASES, MODEL  # noqa: E402

BATCH_SIZE = 96


def main() -> int:
    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY отсутствует в .env / окружении", file=sys.stderr)
        return 1
    catalog = [profile for profile in load_catalog() if profile.source == "organizers"]
    if len(catalog) != 66:
        raise ValueError(f"Ожидалось 66 профилей организаторов, получено {len(catalog)}")

    profiles = {profile.id: _sentences(profile.description) for profile in catalog}
    if any(not sentences for sentences in profiles.values()):
        raise ValueError("Найдено пустое описание без предложений")
    texts = list(dict.fromkeys([*FORMAT_PHRASES.values(),
                                *(sentence for sentences in profiles.values() for sentence in sentences)]))
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=30.0, max_retries=2)
    vectors: dict[str, list[float]] = {}
    for offset in range(0, len(texts), BATCH_SIZE):
        batch = texts[offset:offset + BATCH_SIZE]
        response = client.embeddings.create(model=MODEL, input=batch, encoding_format="float")
        if len(response.data) != len(batch):
            raise ValueError("Ответ API содержит неполный пакет эмбеддингов")
        for item in response.data:
            vectors[batch[item.index]] = [round(value, 4) for value in item.embedding]
        print(f"Готово предложений: {min(offset + len(batch), len(texts))}/{len(texts)}")

    payload = {
        "model": MODEL,
        "source_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
        "formats": {event_type: vectors[phrase] for event_type, phrase in FORMAT_PHRASES.items()},
        "profiles": {profile_id: [
            {"sentence": sentence, "embedding": vectors[sentence]} for sentence in sentences
        ] for profile_id, sentences in profiles.items()},
    }
    temporary = EMBEDDINGS_FILE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
                         encoding="utf-8")
    temporary.replace(EMBEDDINGS_FILE)
    print(f"Сохранено: {len(profiles)} профилей, "
          f"{sum(map(len, profiles.values()))} предложений, {len(FORMAT_PHRASES)} форматов, "
          f"{EMBEDDINGS_FILE.stat().st_size} байт")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
