"""Print total OpenAlex spend recorded in the cost ledger: python -m src.cost"""
from . import config
from .report import ledger_total

if __name__ == "__main__":
    usd, calls = ledger_total(config.DATA_DIR / "cost_ledger.jsonl")
    print(f"Network calls: {calls}; spent (from /rate-limit deltas): {'unknown' if usd is None else f'${usd:.2f}'}")
