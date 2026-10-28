import llm


def test_merge_joins_same_role_turns_and_starts_with_user():
    out = llm._merge([
        {"role": "assistant", "content": "orphan"},
        {"role": "user", "content": "a"},
        {"role": "user", "content": "b"},
        {"role": "assistant", "content": "  "},
    ])
    assert out == [{"role": "user", "content": "a\n\nb"}]


def test_model_for_uses_quick_model_on_anthropic(monkeypatch):
    monkeypatch.setattr(llm, "PROVIDER", "anthropic")
    assert llm.model_for("quick") != llm.model_for("default")
