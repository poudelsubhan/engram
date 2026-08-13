from engram.checkers import (
    check,
    check_contains,
    check_exact,
    check_numeric_tol,
    parse_citations,
)


class TestCitationParsing:
    def test_pulls_ids_out_of_prose(self):
        trace = "Ratings live at imdb.rating [M-0007], so I filtered on it [M-0012]."
        assert parse_citations(trace) == ["M-0007", "M-0012"]

    def test_deduplicates_and_preserves_order(self):
        assert parse_citations("[M-2] then [M-1] then [M-2]") == ["M-2", "M-1"]

    def test_ignores_near_misses(self):
        assert parse_citations("M-0007 and [M0007] and [X-0007] and [M-]") == []

    def test_empty_and_none_are_safe(self):
        assert parse_citations("") == []
        assert parse_citations(None) == []


class TestExact:
    def test_ignores_case_whitespace_and_trailing_period(self):
        assert check_exact("  The Matrix.  ", "the matrix")

    def test_rejects_different_answers(self):
        assert not check_exact("The Matrix", "Fight Club")


class TestNumericTol:
    def test_finds_the_number_inside_a_sentence(self):
        assert check_numeric_tol("There are 42 such movies.", "42")

    def test_handles_thousands_separators(self):
        assert check_numeric_tol("I counted 1,234 movies", "1234")

    def test_rejects_a_wrong_count(self):
        assert not check_numeric_tol("There are 7 such movies.", "42")

    def test_absolute_slack_covers_off_by_rounding_not_off_by_one_on_small_counts(self):
        assert check_numeric_tol("42.4", "42")
        assert not check_numeric_tol("44", "42")

    def test_no_number_in_answer_fails(self):
        assert not check_numeric_tol("I could not determine the count.", "42")

    def test_the_divide_by_ten_lie_produces_a_wrong_count(self):
        """Beat 2 of the poison act depends on this: a /10-normalised threshold
        query returns a count nowhere near the truth."""
        assert not check_numeric_tol("The answer is 0 movies.", "173")


class TestContains:
    def test_all_fragments_must_appear(self):
        answer = "The top 3 are The Matrix, Fight Club, and American Beauty."
        assert check_contains(answer, "The Matrix|Fight Club|American Beauty")

    def test_a_missing_fragment_fails(self):
        answer = "The top 3 are The Matrix and Fight Club."
        assert not check_contains(answer, "The Matrix|Fight Club|American Beauty")

    def test_order_in_the_answer_does_not_matter_only_presence(self):
        assert check_contains("Fight Club, The Matrix", "The Matrix|Fight Club")

    def test_empty_expected_fails_closed(self):
        assert not check_contains("anything", "")


def test_dispatch_falls_back_to_exact_for_unknown_checkers():
    assert check("The Matrix", "the matrix", "no_such_checker")
    assert check("42", "42", "numeric_tol")
    assert check("a|b", "a|b", "contains")
