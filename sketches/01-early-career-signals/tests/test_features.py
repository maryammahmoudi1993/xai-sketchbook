import pandas as pd
import pytest

from src import features


def make():
    early = pd.DataFrame([
        dict(author_id="A1", work_id="W1", cites=10, source_id="S1", topic_id="T1", position="first",
             team_size=3, coauthors=["C1", "C2"], institutions=["I1"]),
        dict(author_id="A1", work_id="W2", cites=5, source_id="S2", topic_id="T2", position="last",
             team_size=2, coauthors=["C1"], institutions=["I1"]),
        dict(author_id="A2", work_id="W3", cites=0, source_id=None, topic_id=None, position="middle",
             team_size=1, coauthors=[], institutions=[]),
    ])
    sources = pd.DataFrame({"source_id": [f"S{i}" for i in range(1, 11)],
                            "name": ["Neural Information Processing Systems"] + ["J%d" % i for i in range(9)],
                            "citedness": [1.0] + [float(i) for i in range(2, 11)]})
    # S1 is NeurIPS (hand-listed, low citedness); S2 has the top citedness only if it were in the data
    co = pd.DataFrame({"author_id": [f"C{i}" for i in range(1, 21)], "h_index": list(range(1, 20)) + [100]})
    inst = pd.DataFrame({"inst_id": [f"I{i}" for i in range(1, 9)], "h_index": [10, 20, 30, 40, 50, 60, 70, 80]})
    cohort = pd.DataFrame({"author_id": ["A1", "A2"], "t0": [2009, 2010]})
    return early, cohort, sources, co, inst


def test_entropy():
    assert features.entropy([1, 1]) == pytest.approx(0.6931, abs=1e-3)
    assert features.entropy([5]) == 0.0


def test_feature_values():
    early, cohort, sources, co, inst = make()
    f, log = features.build_features(early, cohort, sources, co, inst)
    a1 = f.set_index("author_id").loc["A1"]
    assert a1.n_papers_early == 2 and a1.early_citations == 15 and a1.n_distinct_venues == 2
    assert a1.n_unique_coauthors == 2 and a1.mean_team_size == 2.5 and a1.first_author_share == 0.5
    assert a1.n_top_venue_papers >= 1                     # NeurIPS matched via hand list
    assert "NeurIPS" in log["venues"]["hand"] and log["venues"]["hand"]["NeurIPS"]
    assert "ICML" in log["venues"]["hand_unmatched"]
    assert a1.cohort_year == 2009
    a2 = f.set_index("author_id").loc["A2"]
    assert a2.institution_tier == 0 and a2.has_top_coauthor == 0 and a2.topic_diversity == 0


def test_top_coauthor_threshold_and_tiers():
    early, cohort, sources, co, inst = make()
    early.loc[0, "coauthors"] = ["C20"]   # h=100 is above the 95th percentile
    f, _ = features.build_features(early, cohort, sources, co, inst)
    assert f.set_index("author_id").loc["A1", "has_top_coauthor"] == 1
    # I1 has the lowest institution h-index -> tier 4
    assert f.set_index("author_id").loc["A1", "institution_tier"] == 4
