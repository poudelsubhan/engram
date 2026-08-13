"""The extractor is a small model running unattended on every episode.
Its output parser has to survive fenced code blocks, prose, and garbage."""

from engram.llm import parse_memory_json


def test_parses_a_clean_array():
    out = parse_memory_json(
        '[{"text": "imdb.rating is a 0-10 float on the movies collection.", '
        '"kind": "fact"}]'
    )
    assert out == [
        {"text": "imdb.rating is a 0-10 float on the movies collection.",
         "kind": "fact"}
    ]


def test_strips_markdown_fences():
    raw = '```json\n[{"text": "Filter by year before aggregating.", ' \
          '"kind": "procedure"}]\n```'
    assert parse_memory_json(raw)[0]["kind"] == "procedure"


def test_digs_the_array_out_of_surrounding_prose():
    raw = 'Sure! Here are the memories:\n[{"text": "Use $match then $group.", ' \
          '"kind": "procedure"}]\nHope that helps.'
    assert len(parse_memory_json(raw)) == 1


def test_caps_at_three_memories():
    items = ", ".join(
        f'{{"text": "A durable claim number {i} about the schema.", "kind": "fact"}}'
        for i in range(8)
    )
    assert len(parse_memory_json(f"[{items}]")) == 3


def test_drops_trivially_short_claims():
    raw = '[{"text": "yes", "kind": "fact"}, ' \
          '{"text": "Ratings are stored under imdb.rating.", "kind": "fact"}]'
    out = parse_memory_json(raw)
    assert len(out) == 1 and out[0]["text"].startswith("Ratings")


def test_unknown_kind_falls_back_to_fact():
    raw = '[{"text": "Movies have a year field stored as an int.", "kind": "vibes"}]'
    assert parse_memory_json(raw)[0]["kind"] == "fact"


def test_garbage_yields_nothing_rather_than_raising():
    for raw in ("", None, "no json here", "[{broken", '{"text": "not an array"}'):
        assert parse_memory_json(raw) == []


def test_non_dict_entries_are_skipped():
    raw = '["just a string", {"text": "Directors live in the directors array.", ' \
          '"kind": "fact"}]'
    assert len(parse_memory_json(raw)) == 1
