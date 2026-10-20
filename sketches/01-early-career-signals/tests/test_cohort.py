import pandas as pd

from src import cohort


def rows(author, years, cites=1):
    return [{"author_id": author, "work_id": f"{author}-{y}-{i}", "year": y, "cites": cites}
            for i, y in enumerate(years)]


def test_select_cohort_filters_t0_and_min_early_works():
    df = pd.DataFrame(
        rows("A1", [2009, 2010])            # ok: t0 2009, 2 early works
        + rows("A2", [2005, 2009, 2010])    # t0 2005 -> senior, excluded
        + rows("A3", [2009, 2013])          # only 1 work within 2009..2011 -> excluded
        + rows("A4", [2011, 2011, 2013])    # t0 2011, 2 early works -> ok
        + rows("A5", [2012, 2012])          # t0 2012 -> out of range
    )
    t0, log = cohort.select_cohort(df)
    assert set(t0.author_id) == {"A1", "A4"}
    assert t0.set_index("author_id").t0.to_dict() == {"A1": 2009, "A4": 2011}
    assert log["cohort_after_min_early_works"] == 2


def test_outcome_window_and_dedup():
    a = pd.DataFrame(rows("A1", [2009, 2010], cites=5))
    b = pd.DataFrame(rows("A1", [2014, 2023, 2024, 2010], cites=2))  # 2024 = t0+15 -> out; dup 2010 id differs
    t0 = pd.DataFrame({"author_id": ["A1"], "t0": [2009]})
    out = cohort.outcome_works(a, b, t0)
    assert sorted(out.year) == [2009, 2010, 2010, 2014, 2023]
    assert out.work_id.is_unique
