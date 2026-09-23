"""LLM-контракт и отказы проверяются поддельным SDK без сети и расходов."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
from datetime import date
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace

import httpx
import openai
import pytest

from app import explain_llm as llm
from app.data import load_catalog
from app.matcher import Candidate, MatchRequest, _explanation_facts, _score, match


@pytest.fixture
def isolated_cache():
    # Не делим pytest-of-<user> между Windows-процессами с разными правами.
    with TemporaryDirectory(prefix="beyonder-llm-tests-") as directory:
        yield Path(directory) / "explanations.json"


@pytest.fixture
def bundle(monkeypatch, isolated_cache):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("EXPLANATIONS_LLM_ENABLED", "true")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setattr(llm, "CACHE_FILE", isolated_cache)
    llm._MEMORY.clear()

    async def no_network(*args, **kwargs):
        raise AssertionError("Настоящий HTTP запрещён в тестах")

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", no_network)
    catalog = load_catalog()
    req = MatchRequest("Алматы", date(2026, 9, 23), "корпоратив", "Ведущий", 2_000_000)
    cards = match(req, catalog)["cards"]
    by_id = {c.id: c for c in catalog}
    shown = [Candidate(by_id[c["id"]]) for c in cards]
    for cand in shown:
        _score(cand, req)
    pool = [c for c in catalog if req.category in c.categories and c.city == req.city]
    context = {"pool_size": len(pool), "busy_in_pool": sum(req.date in c.busy for c in pool),
               "facts_by_id": {c.c.id: _explanation_facts(c, req, shown) for c in shown}}
    payload = {"request": asdict(req), "context": {k: context[k] for k in ("pool_size", "busy_in_pool")},
               "cards": [context["facts_by_id"][c["id"]] for c in cards]}
    texts = [card["explanation"] for card in cards]
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-never-sent")
    return SimpleNamespace(cards=cards, req=req, context=context, payload=payload,
                           texts=texts, catalog=catalog, raw=json.dumps({"explanations": texts}, ensure_ascii=False))


@pytest.fixture
def fake_sdk(monkeypatch):
    def install(outputs):
        calls, clients = [], []

        class Client:
            def __init__(self, **kwargs):
                clients.append(kwargs)
                self.chat = SimpleNamespace(completions=self)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def create(self, **kwargs):
                calls.append(deepcopy(kwargs))
                item = outputs[min(len(calls) - 1, len(outputs) - 1)]
                if isinstance(item, Exception):
                    raise item
                if callable(item):
                    item = await item()
                return SimpleNamespace(choices=[SimpleNamespace(
                    finish_reason="stop", message=SimpleNamespace(content=item, refusal=None))])

        monkeypatch.setattr(openai, "AsyncOpenAI", Client)
        return calls, clients

    return install


def test_single_batch_strict_json_and_persistent_cache(bundle, fake_sdk):
    b = bundle
    calls, clients = fake_sdk([b.raw])
    assert llm.rewrite(b.cards, b.req, b.context) == b.texts
    llm._MEMORY.clear()  # повторный запуск читает файл
    assert llm.rewrite(b.cards, b.req, b.context) == b.texts
    assert len(calls) == 1
    assert calls[0]["model"] == "test-model"
    assert calls[0]["temperature"] == 0
    assert calls[0]["response_format"]["json_schema"]["strict"] is True
    assert clients[0]["max_retries"] == 0
    assert clients[0]["timeout"] == 6.0
    facts = json.loads(calls[0]["messages"][1]["content"])
    assert len(facts["cards"]) == len(b.cards) == 3
    assert all("comparisons" in c and "score" not in c and "name" not in c for c in facts["cards"])
    cache = json.loads(llm.CACHE_FILE.read_text(encoding="utf-8"))
    assert list(cache) == [llm._cache_key(asdict(b.req), b.cards)]
    assert len(next(iter(cache))) == 64


def test_no_key_or_disabled_never_calls_sdk(bundle, fake_sdk, monkeypatch):
    calls, _ = fake_sdk([bundle.raw])
    monkeypatch.delenv("OPENAI_API_KEY")
    assert llm.rewrite(bundle.cards, bundle.req, bundle.context) is None
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-never-sent")
    monkeypatch.setenv("EXPLANATIONS_LLM_ENABLED", "false")
    assert llm.rewrite(bundle.cards, bundle.req, bundle.context) is None
    assert calls == []


def test_nvidia_provider_uses_compatible_endpoint_and_separate_cache(bundle, fake_sdk, monkeypatch):
    calls, clients = fake_sdk([bundle.raw])
    monkeypatch.setenv("LLM_PROVIDER", "nvidia")
    monkeypatch.setenv("NVIDIA_API_KEY", "test-only-never-sent")
    monkeypatch.setenv("NVIDIA_MODEL", "test-model")
    assert llm.rewrite(bundle.cards, bundle.req, bundle.context) == bundle.texts
    assert clients[0]["base_url"] == "https://integrate.api.nvidia.com/v1"
    assert calls[0]["max_tokens"] == 1200
    assert "response_format" not in calls[0]
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert llm.rewrite(bundle.cards, bundle.req, bundle.context) == bundle.texts
    assert len(calls) == 2  # одинаковая модель другого провайдера не даёт cache hit


def test_nvidia_missing_key_or_model_stays_on_templates(bundle, fake_sdk, monkeypatch):
    calls, _ = fake_sdk([bundle.raw])
    monkeypatch.setenv("LLM_PROVIDER", "nvidia")
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("NVIDIA_MODEL", "test-model")
    assert llm.rewrite(bundle.cards, bundle.req, bundle.context) is None
    monkeypatch.setenv("NVIDIA_API_KEY", "test-only-never-sent")
    monkeypatch.delenv("NVIDIA_MODEL", raising=False)
    assert llm.rewrite(bundle.cards, bundle.req, bundle.context) is None
    assert calls == []


def test_validation_retry_includes_errors(bundle, fake_sdk):
    calls, _ = fake_sdk(["not JSON", bundle.raw])
    assert llm.rewrite(bundle.cards, bundle.req, bundle.context) == bundle.texts
    assert len(calls) == 2
    assert "Ошибки проверки" in calls[1]["messages"][-1]["content"]
    assert "корректным JSON" in calls[1]["messages"][-1]["content"]


def test_two_invalid_responses_pin_template_fallback(bundle, fake_sdk):
    calls, _ = fake_sdk(['{"explanations": []}'])
    assert llm.rewrite(bundle.cards, bundle.req, bundle.context) is None
    assert llm.rewrite(bundle.cards, bundle.req, bundle.context) is None
    assert len(calls) == 2
    assert next(iter(json.loads(llm.CACHE_FILE.read_text(encoding="utf-8")).values()))["source"] == "template"


def test_api_error_does_not_retry_and_match_still_works(bundle, fake_sdk):
    calls, _ = fake_sdk([RuntimeError("service unavailable")])
    response = match(bundle.req, bundle.catalog)
    assert response["cards"] == bundle.cards
    assert len(calls) == 1


def test_match_marks_llm_source_without_changing_order(bundle, fake_sdk):
    fake_sdk([bundle.raw])
    result = match(bundle.req, bundle.catalog)
    assert [c["id"] for c in result["cards"]] == [c["id"] for c in bundle.cards]
    assert [c["explanation"] for c in result["cards"]] == bundle.texts
    assert all(c["explanation_source"] == "llm" for c in result["cards"])


@pytest.mark.parametrize("bad", [
    "Отличный выбор для мероприятия на 23 сентября.",
    "Профессионал своего дела за 1000000.",
    "Цена от 1000000. Есть русский. Есть казахский.",
    "Свободен и работает на вашем языке.",
    "Цена от 987654321 ₸.",
    "В описании: «Лауреат несуществующей премии».",
    "",
])
def test_validator_rejects_ungrounded_or_generic_text(bundle, bad):
    texts = [bad, *bundle.texts[1:]]
    value, errors = llm._validate(json.dumps({"explanations": texts}), bundle.payload, bundle.cards)
    assert value is None and errors


def test_validator_rejects_duplicates_and_extra_fields(bundle):
    value, errors = llm._validate(json.dumps({"explanations": [bundle.texts[0]] * 3}), bundle.payload, bundle.cards)
    assert value is None
    assert any("совпадают" in error for error in errors)
    assert llm._validate(json.dumps({"explanations": bundle.texts, "extra": 1}), bundle.payload, bundle.cards)[1]


@pytest.mark.parametrize(("text", "expected_error"), [
    ("Цена от 1 300 000 ₸. В описании — «работает с крупнейшими брендами».", "начни с причины"),
    ("В описании — «работает с крупнейшими брендами». Свободен 23 сентября; цена от 1 300 000 ₸.", "общая занятость"),
    ("В описании — «работает с крупнейшими брендами». Цена от 1 300 000 ₸. " + "Очень " * 40, "220 символов"),
])
def test_validator_enforces_v2_presentation(bundle, text, expected_error):
    texts = [text, *bundle.texts[1:]]
    value, errors = llm._validate(json.dumps({"explanations": texts}), bundle.payload, bundle.cards)
    assert value is None
    assert any(expected_error in error for error in errors)


@pytest.mark.parametrize("pronoun", ["он", "Она"])
def test_validator_rejects_gendered_pronouns_outside_quotes(bundle, pronoun):
    texts = [bundle.texts[0] + f" {pronoun} берёт этот формат.", *bundle.texts[1:]]
    value, errors = llm._validate(json.dumps({"explanations": texts}), bundle.payload, bundle.cards)
    assert value is None
    assert any("не угадывай род" in error for error in errors)


@pytest.mark.parametrize("quote", [
    "Ведёт свадьбы на казахском языке",
    "Она ведёт свадьбы на казахском языке",
    "Он ведёт свадьбы на казахском языке",
])
def test_validator_accepts_description_quote_before_price(quote):
    cards = [{"id": "one", "name": "Имя"}]
    payload = {"request": {}, "context": {}, "cards": [{"price_from_kzt": 100,
               "comparisons": [], "description_quote": quote}]}
    raw = json.dumps({"explanations": [f"В описании: «{quote}». Цена от 100 ₸."]})
    assert llm._validate(raw, payload, cards)[1] == []


def test_total_timeout_includes_validation_retry(bundle, fake_sdk, monkeypatch):
    async def first():
        await asyncio.sleep(0.03)
        return "bad json"

    async def second():
        await asyncio.sleep(1)
        return bundle.raw

    monkeypatch.setattr(llm, "TIMEOUT_SECONDS", 0.12)
    calls, _ = fake_sdk([first, second])
    start = time.monotonic()
    assert llm.rewrite(bundle.cards, bundle.req, bundle.context) is None
    assert time.monotonic() - start < 0.75
    assert len(calls) == 2


def test_concurrent_same_query_has_single_generation(bundle, fake_sdk):
    async def slow():
        await asyncio.sleep(0.04)
        return bundle.raw

    calls, _ = fake_sdk([slow])
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: llm.rewrite(bundle.cards, bundle.req, bundle.context), range(2)))
    assert results == [bundle.texts, bundle.texts]
    assert len(calls) == 1


def test_cache_invalidates_for_prompt_model_and_facts(bundle, fake_sdk, monkeypatch):
    calls, _ = fake_sdk([bundle.raw])
    assert llm.rewrite(bundle.cards, bundle.req, bundle.context) == bundle.texts
    monkeypatch.setattr(llm, "PROMPT_VERSION", "next-prompt")
    assert llm.rewrite(bundle.cards, bundle.req, bundle.context) == bundle.texts
    monkeypatch.setenv("OPENAI_MODEL", "other-model")
    assert llm.rewrite(bundle.cards, bundle.req, bundle.context) == bundle.texts
    changed = deepcopy(bundle.context)
    changed["busy_in_pool"] += 1
    assert llm.rewrite(bundle.cards, bundle.req, changed) == bundle.texts
    assert len(calls) == 4


def test_corrupt_cache_and_failed_disk_write_do_not_break_matching(bundle, fake_sdk, monkeypatch):
    llm.CACHE_FILE.write_text("broken JSON", encoding="utf-8")
    calls, _ = fake_sdk([bundle.raw])

    def cannot_replace(*args):
        raise OSError("read-only disk")

    monkeypatch.setattr(llm.os, "replace", cannot_replace)
    assert llm.rewrite(bundle.cards, bundle.req, bundle.context) == bundle.texts
    assert llm.rewrite(bundle.cards, bundle.req, bundle.context) == bundle.texts
    assert len(calls) == 1


def test_real_sdk_serializes_batch_against_mock_transport(bundle, monkeypatch):
    real_client = openai.AsyncOpenAI
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(200, json={
            "id": "chatcmpl-local-test", "object": "chat.completion", "created": 0,
            "model": "test-model", "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": bundle.raw, "refusal": None}}],
        })

    def client_factory(**kwargs):
        return real_client(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    monkeypatch.setattr(openai, "AsyncOpenAI", client_factory)
    assert llm.rewrite(bundle.cards, bundle.req, bundle.context) == bundle.texts
    assert len(requests) == 1
    assert requests[0]["temperature"] == 0
    schema = requests[0]["response_format"]["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["explanations"]["minItems"] == 3
