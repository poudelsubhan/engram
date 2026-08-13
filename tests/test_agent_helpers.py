"""Pure pieces of the agent loop: how memories are shown to the model, how the
final answer is lifted out of a trace, and which citations are allowed to count."""

from engram import checkers
from engram.agent import _final_answer, render_memories


class TestRenderMemories:
    def test_matches_the_format_the_system_prompt_promises(self):
        docs = [{"mid": "M-0007", "trust": 0.7231,
                 "text": "In sample_mflix, ratings live at imdb.rating on a 0-10 scale."}]
        assert render_memories(docs) == (
            "MEMORY [M-0007] (trust 0.72): In sample_mflix, ratings live at "
            "imdb.rating on a 0-10 scale."
        )

    def test_rendered_ids_are_parseable_back_out(self):
        docs = [{"mid": "M-0001", "trust": 0.3, "text": "a claim about the schema"},
                {"mid": "M-0002", "trust": 0.9, "text": "another claim"}]
        assert checkers.parse_citations(render_memories(docs)) == ["M-0001", "M-0002"]

    def test_empty_population_says_so_rather_than_rendering_nothing(self):
        assert "no memories yet" in render_memories([])


class TestFinalAnswer:
    def test_lifts_the_marked_line(self):
        trace = "I queried the collection.\nFINAL ANSWER: 42"
        assert _final_answer(trace) == "42"

    def test_takes_the_last_marker_when_the_model_repeats_itself(self):
        trace = "FINAL ANSWER: 7\nwait, let me recount.\nFINAL ANSWER: 42"
        assert _final_answer(trace) == "42"

    def test_falls_back_to_the_last_line_when_the_model_forgets_the_marker(self):
        assert _final_answer("some reasoning\n42") == "42"

    def test_empty_trace_is_safe(self):
        assert _final_answer("") == ""


class TestCitationGating:
    """`record` only counts citations of memories actually served this episode —
    otherwise a hallucinated [M-9999] would create trust signal from nothing."""

    @staticmethod
    def _gate(trace, retrieved):
        return [m for m in checkers.parse_citations(trace) if m in retrieved]

    def test_citations_of_served_memories_count(self):
        assert self._gate("used [M-0001] here", ["M-0001", "M-0002"]) == ["M-0001"]

    def test_invented_ids_are_discarded(self):
        assert self._gate("as per [M-9999]", ["M-0001"]) == []

    def test_a_run_with_no_memories_can_have_no_citations(self):
        assert self._gate("I cited [M-0001] anyway", []) == []
