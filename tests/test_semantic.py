"""Предрассчитанный индекс используется офлайн и не зависит от ключа API."""
from datetime import date
import hashlib
import json
import math
from pathlib import Path

from app import semantic
from app.data import DATASET, load_catalog
from app.matcher import Candidate, MatchRequest, _score, _sentences, match


def test_index_contains_every_sentence_and_six_format_vectors():
    payload = json.loads(semantic.EMBEDDINGS_FILE.read_text(encoding="utf-8"))
    catalog = [p for p in load_catalog() if p.source == "organizers"]
    assert len(catalog) == len(payload["profiles"]) == 66
    assert payload["model"] == "text-embedding-3-small"
    assert payload["source_sha256"] == hashlib.sha256(DATASET.read_bytes()).hexdigest()
    assert set(payload["formats"]) == set(semantic.FORMAT_PHRASES)
    dimensions = len(next(iter(payload["formats"].values())))
    assert dimensions > 0
    assert all(value == round(value, 4) for vector in payload["formats"].values()
               for value in vector)
    for profile in catalog:
        entries = payload["profiles"][profile.id]
        assert [entry["sentence"] for entry in entries] == _sentences(profile.description)
        assert all(len(entry["embedding"]) == dimensions for entry in entries)
        assert all(value == round(value, 4) for entry in entries
                   for value in entry["embedding"])


def test_missing_index_enables_fallback():
    missing = Path(__file__).parent / "__no_embedded_catalog_for_test__.json"
    assert not missing.exists()
    assert semantic.load_index(missing) is None


def test_relevance_is_best_sentence_cosine_and_fallback_uses_markers(monkeypatch):
    profile = next(p for p in load_catalog() if p.id == "HK-30583")
    req = MatchRequest("Алматы", date(2026, 10, 10), "свадьба", "Фотограф", 800_000)
    candidate = Candidate(profile)
    _score(candidate, req)
    ranked = semantic.ranked_sentences(semantic.INDEX, profile.id,
                                       profile.description, req.event_type)
    assert candidate.snippet == ranked[0][0]
    assert math.isclose(candidate.parts["semantic"], ranked[0][1], abs_tol=1e-12)
    assert candidate.parts["relevance"] == candidate.parts["semantic"]
    assert candidate.marker_hits == 0

    monkeypatch.setattr(semantic, "INDEX", None)
    fallback = Candidate(profile)
    _score(fallback, req)
    assert fallback.parts["semantic"] == 0.0
    assert fallback.marker_hits > 0
    assert fallback.parts["relevance"] != candidate.parts["relevance"]


def test_semantic_order_is_deterministic_without_key_or_network(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("EXPLANATIONS_LLM_ENABLED", "false")
    req = MatchRequest("Алматы", date(2026, 10, 10), "свадьба", "Фотограф", 800_000)
    catalog = load_catalog()
    first = match(req, catalog)
    assert first == match(req, catalog)
    assert [card["id"] for card in first["cards"]] == ["HK-30583", "HK-53108", "HK-16628"]
    assert all(card["score_parts"]["semantic"] > 0 for card in first["cards"])
    assert all(card["explanation_source"] == "template" for card in first["cards"])
