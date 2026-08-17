
from pathlib import Path
import os

# ── Root data directory ──────────────────────────────────────────────
# Default is ./data relative to the project root.
# Override with an environment variable for different machines / CI.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
 
# ── Root data directory ──────────────────────────────────────────────
# Default is <project_root>/data.
# Override with an environment variable for different machines / CI.
DATA_DIR = Path(os.environ.get("EMBEDDING_PROJECT_DATA_DIR", PROJECT_ROOT / "data"))

#%%
# print(f"Using data directory: {DATA_DIR}")

# ── Clients, models, input types ─────────────────────────────────────
CLIENTS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
ClientsNames = ["Client 1 name", "Client 2 name", "Client 3 name", 
                "Client 4 name", "Client 5 name", "Client 6 name", "Client 7 name"]


MODELS = ["SBERT", "mean_DEBERTA", "cls_DEBERTA"]
INPUT_TYPES = ["A", "B", "C"]


# ── Path helpers ─────────────────────────────────────────────────────
# These are thin functions so that every module builds paths the same
# way. 

def client_dir(client: str) -> Path:
    """Return the directory for a given client, e.g. data/C1/."""
    return DATA_DIR / client


def consolidated_csv(client: str) -> Path:
    """e.g. data/C1/C1_consolidated.csv"""
    return client_dir(client) / f"{client}_consolidated.csv"


def merged_csv(client: str) -> Path:
    """e.g. data/C1/C1_merged.csv"""
    return client_dir(client) / f"{client}_merged.csv"


def embeddings_pkl(client: str, input_type: str, model: str) -> Path:
    """e.g. data/C1/C1_embeddings_B_SBERT.pkl"""
    return client_dir(client) / f"{client}_embeddings_{input_type}_{model}.pkl"


def input_csv(client: str, input_type: str) -> Path:
    """e.g. data/C1/C1_input_A.csv"""
    return client_dir(client) / f"{client}_input_{input_type}.csv"