"""Central configuration: paths, cohort definition, budget and hand-listed venues."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("ECS_DATA_DIR", ROOT / "data"))  # gitignored by repo .gitignore
CACHE_DIR = DATA_DIR / "cache"
PROCESSED_DIR = DATA_DIR / "processed"
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "outputs"

SEED = 42
DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"
DRY_RUN_AUTHORS = 300
DRY_RUN_SEED_WORKS = 200

# --- API ---
API_BASE = "https://api.openalex.org"
API_KEY_ENV = "OPENALEX_API_KEY"
MAILTO = os.environ.get("OPENALEX_MAILTO", "mahmoudi.maryam.msc@gmail.com")
MAX_REQ_PER_SEC = 5.0
BATCH_IDS = 50
PER_PAGE = 100
MIN_BUDGET_USD = 0.05  # stop (resumably) when remaining daily budget falls below this

# --- Cohort ---
SUBFIELD_ID = 1702  # Artificial Intelligence
SEED_YEARS = (2008, 2011)
WORK_TYPES = "article|proceedings-article"
T0_RANGE = (2008, 2011)
MIN_EARLY_WORKS = 2
EARLY_WINDOW = 3          # t0..t0+2
OUTCOME_WINDOW = 15       # t0..t0+14
LAST_YEAR = 2025          # t0=2011 -> 2025
SEED_SAMPLE_WORKS = 10000  # per seed-year sample (OpenAlex `sample`, max 10k)
CANDIDATE_AUTHOR_CAP = 40000   # cap on candidate authors screened (budget lever)
TARGET_COHORT = (5000, 10000)
TOP_FRAC = 0.05

# --- Features ---
TOP_VENUE_QUANTILE = 0.90
TOP_COAUTHOR_QUANTILE = 0.95
HAND_VENUES = {
    "NeurIPS": r"neural information processing systems|\bneurips\b|\bnips\b",
    "ICML": r"international conference on machine learning|\bicml\b",
    "ICLR": r"international conference on learning representations|\biclr\b",
    "CVPR": r"computer vision and pattern recognition|\bcvpr\b",
    "ICCV": r"international conference on computer vision|\biccv\b",
    "ECCV": r"european conference on computer vision|\beccv\b",
    "AAAI": r"\baaai\b|association for the advancement of artificial intelligence",
    "IJCAI": r"international joint conference on artificial intelligence|\bijcai\b",
    "ACL": r"annual meeting of the association for computational linguistics|\bacl\b",
    "EMNLP": r"empirical methods in natural language processing|\bemnlp\b",
    "TPAMI": r"pattern analysis and machine intelligence|\btpami\b",
    "JMLR": r"journal of machine learning research|\bjmlr\b",
}
