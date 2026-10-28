import rag


def test_chunk_keeps_whole_sentences_and_overlaps():
    chunks = rag.chunk("One. Two is here. Three!", target_chars=10)
    assert chunks == ["One. Two is here.", "Two is here. Three!"]


def test_chunk_handles_short_text():
    assert rag.chunk("Just one sentence.") == ["Just one sentence."]


def test_parse_rejoins_hyphenated_words():
    assert rag.parse("a.txt", b"mito-\nchondria are cool") == "mitochondria are cool"


def test_tokens_drop_stopwords():
    assert rag._tokens("How does the Cas9 find its PAM?") == ["cas9", "find", "pam"]
