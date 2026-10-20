import pandas as pd

from src import target


def test_h_index():
    assert target.h_index([10, 8, 5, 4, 3]) == 4
    assert target.h_index([]) == 0
    assert target.h_index([0, 0]) == 0


def test_targets_keep_authors_without_late_works():
    cohort = pd.DataFrame({"author_id": [f"A{i}" for i in range(100)], "t0": 2009})
    out = pd.DataFrame({"author_id": [f"A{i}" for i in range(50)], "work_id": range(50),
                        "year": 2010, "cites": list(range(50))})
    t = target.build_targets(out, cohort)
    assert len(t) == 100 and (t.citations_15.iloc[50:] == 0).all()
    assert t.label.sum() >= 5 and t.loc[t.label == 1, "citations_15"].min() >= 45
