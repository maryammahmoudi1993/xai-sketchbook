"""End-to-end orchestration. `python -m src.pipeline [--dry-run]`.

Resumable: every API response is cached on disk, so re-running after a BudgetExhausted stop
(or on another day) only spends budget on requests that were not yet made.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import cohort as co
from . import config, enrich, explain, features, modeling, plots, report, target
from .api import BudgetExhausted, OpenAlexClient


def build_dataset(client: OpenAlexClient, dry_run: bool):
    """Stages 0-B + enrichment + features. Returns (feature/target table, logs)."""
    cands = co.collect_candidate_authors(client, dry_run)
    stage_a = co.fetch_stage_a(client, cands)
    t0, funnel = co.select_cohort(stage_a)
    funnel["candidates_screened"] = len(cands)
    if len(t0) < config.TARGET_COHORT[0]:
        print(f"WARNING: cohort size {len(t0)} is below the {config.TARGET_COHORT[0]} target "
              "(raise CANDIDATE_AUTHOR_CAP / SEED_SAMPLE_WORKS or run on another day).")
    stage_b = co.fetch_stage_b(client, list(t0.author_id))
    outcome = co.outcome_works(stage_a, stage_b, t0)
    targets = target.build_targets(outcome, t0)

    d = stage_a.drop_duplicates(["author_id", "work_id"]).merge(t0[["author_id", "t0"]], on="author_id")
    early = d[(d.year >= d.t0) & (d.year <= d.t0 + config.EARLY_WINDOW - 1)]
    ids = enrich.early_lookup_ids(early)
    sources = enrich.fetch_sources(client, ids["sources"])
    coauth = enrich.fetch_authors(client, ids["authors"])
    insts = enrich.fetch_institutions(client, ids["institutions"])
    feats, flog = features.build_features(early, t0, sources, coauth, insts)
    table = feats.merge(targets[["author_id", "citations_15", "n_papers_15", "h_index_15", "label"]], on="author_id")
    late_ids = set(outcome[outcome.year >= outcome.t0 + config.EARLY_WINDOW].author_id)
    stopped = 1 - table.author_id.isin(late_ids).mean()
    return table, {"funnel": funnel, "feat_log": flog, "threshold": targets.attrs["top5_threshold"],
                   "share_stopped": float(stopped)}


def analyse(table: pd.DataFrame, logs: dict, out_dir: Path, dry_run: bool) -> dict | None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tr, te = modeling.temporal_split(table)
    if min(tr.label.nunique(), te.label.nunique()) < 2:
        print("Not enough positives in train/test to model (expected in a tiny dry run). Skipping modeling.")
        return None
    n_boot = 100 if dry_run else 1000
    res = modeling.run_all(table, n_boot=n_boot)
    X = res["test"][features.FEATURES_NO_EARLY_CITES]
    sv, iv = explain.shap_explain(res["models"]["without_early_citations"], X)
    ga = explain.group_analysis(table, n_boot=300 if dry_run else 2000)
    adj = explain.adjusted_coauthor_effect(table, n_boot=50 if dry_run else 200)
    lg = res["metrics"][("lightgbm", "without_early_citations")]

    plots.fig_group_bars(ga, fig_dir / "1_top_coauthor_by_institution.png")
    plots.fig_shap_beeswarm(sv, X, fig_dir / "2_shap_beeswarm.png")
    plots.fig_interaction(explain.interaction_table(iv, X), fig_dir / "3_interaction.png")
    plots.fig_metrics_card(lg, lg["base_rate"], res["n_test"], len(table), fig_dir / "4_metrics_card.png")
    plots.fig_dependence(sv, X, "has_top_coauthor", fig_dir / "5_dependence_top_coauthor.png")
    plots.fig_dependence(sv, X, "n_papers_early", fig_dir / "6_dependence_n_papers.png")
    plots.fig_calibration(res["calibration"], fig_dir / "7_calibration.png")

    ctx = {"res": res, "ga": ga, "adj": adj, "li": explain.li2019_agreement(adj[1], adj[2], adj[0]),
           "run_date": date.today().isoformat(), "dry_run": dry_run, "n_cohort": len(table),
           "n_pos": int(table.label.sum()), "h_med": float(table.h_index_15.median()),
           "h_p95": float(table.h_index_15.quantile(0.95)), **logs}
    report.write_results(out_dir / "results.md", ctx)
    report.write_limitations(out_dir / "limitations.md", ctx)
    return ctx


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help=f"~{config.DRY_RUN_AUTHORS} candidate authors")
    ap.add_argument("--from-cache-only", action="store_true", help="never call the network (fails on cache miss)")
    args = ap.parse_args(argv)
    dry = args.dry_run or config.DRY_RUN
    out_dir = (config.RESULTS_DIR / "dry_run") if dry else config.ROOT
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    client = OpenAlexClient()
    if args.from_cache_only:
        client.api_key = None
    else:
        rem = client.check_budget(verbose=True)
        print(f"Remaining budget before run: {rem}")
    try:
        table, logs = build_dataset(client, dry)
    except BudgetExhausted as e:
        print(f"STOPPED (resumable): {e}")
        return 2
    finally:
        _log_cost(client, dry)
    table.to_parquet(config.PROCESSED_DIR / ("features_dry.parquet" if dry else "features.parquet"))
    out_dir.mkdir(parents=True, exist_ok=True)
    features.write_dictionary(out_dir / "data_dictionary.csv")
    (config.PROCESSED_DIR / "logs.json").write_text(json.dumps(logs, indent=2, default=str))
    print(f"Cohort: {len(table)} authors, {int(table.label.sum())} top-5%.")
    analyse(table, logs, out_dir, dry)
    return 0


def _log_cost(client: OpenAlexClient, dry: bool) -> None:
    try:
        if client.api_key:
            client.check_budget()
    except BudgetExhausted:
        pass
    except Exception as e:  # cost logging must never mask the real error
        print("cost log skipped:", e)
    rec = {"time": datetime.now().isoformat(timespec="seconds"), "dry_run": dry,
           "network_calls": client.network_calls, "cache_hits": client.cache_hits,
           "spent_usd": client.spent_this_session()}
    with open(config.DATA_DIR / "cost_ledger.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("Run cost:", rec)


if __name__ == "__main__":
    raise SystemExit(main())
