"""The agent gets one tool and it is read-only. These tests fail closed:
every rejection is checked without ever reaching a database."""

import pytest

from engram import mongo_tool
from engram.mongo_tool import QueryRejected, _scan_stages, run_mongo_query


class TestStageScanner:
    @pytest.mark.parametrize("stage", ["$out", "$merge", "$function", "$where",
                                       "$accumulator"])
    def test_write_and_execution_stages_are_rejected(self, stage):
        with pytest.raises(QueryRejected):
            _scan_stages([{"$match": {"year": 1995}}, {stage: "anything"}])

    def test_cross_collection_stages_are_rejected(self):
        for stage in ("$lookup", "$unionWith", "$graphLookup"):
            with pytest.raises(QueryRejected):
                _scan_stages([{stage: {}}])

    def test_ordinary_read_stages_pass(self):
        _scan_stages([
            {"$match": {"year": 1995, "imdb.rating": {"$gt": 8}}},
            {"$group": {"_id": None, "n": {"$sum": 1}}},
            {"$sort": {"n": -1}},
            {"$limit": 3},
        ])

    def test_non_object_stages_are_rejected(self):
        with pytest.raises(QueryRejected):
            _scan_stages(["$match"])


class TestRequestValidation:
    """These reject before any driver call, so they run without a cluster."""

    def test_unknown_op_is_rejected(self):
        out = run_mongo_query({"op": "update", "filter": {}})
        assert out["ok"] is False and "op must be one of" in out["error"]

    def test_delete_op_is_rejected(self):
        assert run_mongo_query({"op": "deleteMany"})["ok"] is False

    def test_unknown_collection_is_rejected(self):
        out = run_mongo_query({"op": "find", "collection": "engram_memories"})
        assert out["ok"] is False and "Unknown collection" in out["error"]

    def test_forbidden_stage_is_rejected_before_execution(self):
        out = run_mongo_query({
            "op": "aggregate", "collection": "movies",
            "pipeline": [{"$out": "stolen"}],
        })
        assert out["ok"] is False and "$out is not allowed" in out["error"]

    def test_non_list_pipeline_is_rejected(self):
        out = run_mongo_query({"op": "aggregate", "pipeline": {"$match": {}}})
        assert out["ok"] is False and "array of stages" in out["error"]

    def test_a_json_string_payload_is_accepted_and_still_validated(self):
        out = run_mongo_query('{"op": "drop"}')
        assert out["ok"] is False and "op must be one of" in out["error"]

    def test_malformed_json_never_raises(self):
        assert run_mongo_query("{not json")["ok"] is False

    def test_a_bare_list_is_rejected(self):
        assert run_mongo_query("[1,2,3]")["ok"] is False


def test_tool_schema_advertises_only_read_operations():
    params = mongo_tool.TOOL_SCHEMA["function"]["parameters"]
    assert mongo_tool.TOOL_SCHEMA["function"]["name"] == "run_mongo_query"
    assert set(params["properties"]["op"]["enum"]) == {"find", "aggregate"}
    assert params["required"] == ["op"]


def test_document_cap_is_enforced_at_25():
    assert mongo_tool.MAX_DOCS == 25
