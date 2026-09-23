"""Загрузка каталога подрядчиков из CSV организаторов."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATASET = DATA_DIR / "hackathon-dataset-anonymized.csv"
EXTRA_DATASET = DATA_DIR / "synthetic-extra.csv"  # наши синтетические профили (опционально)

CALENDAR_START = date(2026, 9, 23)
CALENDAR_END = date(2026, 12, 31)


@dataclass(frozen=True)
class Contractor:
    id: str
    name: str
    categories: tuple[str, ...]
    city: str
    city_imputed: bool
    synthetic: bool
    price: int
    price_imputed: bool
    formats: frozenset[str]
    languages: frozenset[str]
    max_hours: int | None
    busy: frozenset[date]
    description: str
    source: str  # "organizers" | "team"


def _bool(v: str) -> bool:
    return v.strip().lower() in ("true", "1", "yes")


def _split(v: str) -> list[str]:
    return [x.strip() for x in v.split("|") if x.strip()]


def _load_file(path: Path, source: str) -> list[Contractor]:
    out: list[Contractor] = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            out.append(
                Contractor(
                    id=r["id"].strip(),
                    name=r["anon_name"].strip(),
                    categories=tuple(_split(r["categories"])),
                    city=r["city"].strip(),
                    city_imputed=_bool(r["city_imputed"]),
                    synthetic=_bool(r["synthetic"]),
                    price=int(float(r["price_from_kzt"])),
                    price_imputed=_bool(r["price_imputed"]),
                    formats=frozenset(_split(r["event_formats"])),
                    languages=frozenset(_split(r["languages"])),
                    max_hours=int(float(r["max_hours"])) if r["max_hours"].strip() else None,
                    busy=frozenset(date.fromisoformat(d) for d in _split(r["busy_dates"])),
                    description=r["description"].strip(),
                    source=source,
                )
            )
    return out


def load_catalog() -> list[Contractor]:
    items = _load_file(DATASET, "organizers")
    if EXTRA_DATASET.exists():
        items += _load_file(EXTRA_DATASET, "team")
    items.sort(key=lambda c: c.id)
    return items
