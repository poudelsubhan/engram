"""These two pipelines are the MongoDB-fluency exhibit and they go in the README
verbatim. Pin their shape so a refactor can't quietly change what's advertised."""

from engram import config, store
from engram import trust as T


class TestRetrievalPipeline:
    def setup_method(self):
        self.p = store.retrieval_pipeline([0.1, 0.2, 0.3], "how are ratings stored?", k=4)

    def test_vector_search_is_the_first_stage(self):
        assert list(self.p[0]) == ["$vectorSearch"]

    def test_search_stage_targets_the_named_index_and_path(self):
        s = self.p[0]["$vectorSearch"]
        assert s["index"] == config.VECTOR_INDEX == "mem_vec"
        assert s["path"] == "embedding"
        assert s["numCandidates"] == 50 and s["limit"] == 12
        assert s["queryVector"] == [0.1, 0.2, 0.3]
        assert "query" not in s

    def test_quarantined_and_dead_memories_never_surface(self):
        match = next(st["$match"] for st in self.p if "$match" in st)
        assert set(match["status"]["$nin"]) == {T.QUARANTINED, T.DEAD}

    def test_quarantined_memories_are_also_filtered_inside_the_index(self):
        """Without the pre-filter, quarantined memories would still consume
        candidate slots and starve healthy ones out of the top 12."""
        f = self.p[0]["$vectorSearch"]["filter"]
        assert set(f["status"]["$nin"]) == {T.QUARANTINED, T.DEAD}

    def test_score_is_similarity_scaled_by_earned_trust(self):
        stage = next(
            st["$addFields"] for st in self.p
            if "$addFields" in st and "score" in st["$addFields"]
        )
        assert stage["score"] == {
            "$multiply": [
                "$vectorScore",
                {"$add": [0.4, {"$multiply": [0.6, "$trust"]}]},
            ]
        }

    def test_vector_score_is_captured_from_meta(self):
        stage = self.p[1]["$addFields"]
        assert stage["vectorScore"] == {"$meta": "vectorSearchScore"}

    def test_sorts_by_trust_weighted_score_then_limits_to_k(self):
        assert {"$sort": {"score": -1}} in self.p
        assert {"$limit": 4} in self.p
        assert self.p.index({"$sort": {"score": -1}}) < self.p.index({"$limit": 4})

    def test_embeddings_are_not_shipped_back_to_the_client(self):
        assert self.p[-1] == {"$project": {"embedding": 0}}

    def test_auto_embedding_mode_sends_text_instead_of_a_vector(self):
        p = store.retrieval_pipeline(None, "how are ratings stored?", k=4)
        s = p[0]["$vectorSearch"]
        assert s["query"] == "how are ratings stored?"
        assert "queryVector" not in s

    def test_k_is_honoured(self):
        assert {"$limit": 2} in store.retrieval_pipeline([0.0], "q", k=2)


class TestContaminationPipeline:
    def setup_method(self):
        self.p = store.contamination_pipeline("M-0007")

    def test_starts_from_the_quarantined_memory(self):
        assert self.p[0] == {"$match": {"mid": "M-0007"}}

    def test_graph_lookup_walks_mid_into_parents(self):
        g = self.p[1]["$graphLookup"]
        assert g["from"] == "memories"
        assert g["startWith"] == "$mid"
        assert g["connectFromField"] == "mid"
        assert g["connectToField"] == "parents"
        assert g["as"] == "descendants"
        assert g["depthField"] == "depth"

    def test_traversal_is_bounded_so_a_provenance_cycle_cannot_hang_the_demo(self):
        assert self.p[1]["$graphLookup"]["maxDepth"] == config.MAX_CASCADE_DEPTH == 5

    def test_dead_memories_are_not_resurrected_by_the_walk(self):
        assert self.p[1]["$graphLookup"]["restrictSearchWithMatch"] == {
            "status": {"$ne": T.DEAD}
        }

    def test_it_is_a_single_round_trip(self):
        assert sum("$graphLookup" in st for st in self.p) == 1
        assert len(self.p) == 3
