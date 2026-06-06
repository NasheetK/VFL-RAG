#!/usr/bin/env python3
"""Patch Train.ipynb for 5-class CIC2017-only training."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
nb_path = ROOT / "Train.ipynb"
nb = json.loads(nb_path.read_text(encoding="utf-8"))

OLD = '''# 1. Load Dataset & Extract Multi-Class Labels
# -----------------------------
# Dataset folder path
DATASETS_FOLDER = Path("datasets")

# Find all CSV files in the datasets folder (only standard .csv files)
csv_files = [f for f in DATASETS_FOLDER.glob("*.csv") if f.suffix == ".csv"]

if len(csv_files) == 0:
    raise FileNotFoundError(f"No CSV files found in {DATASETS_FOLDER}")

print(f"Found {len(csv_files)} CSV file(s) in {DATASETS_FOLDER}:")
for csv_file in csv_files:
    print(f"  - {csv_file.name}")

# Load all CSV files and concatenate them
print(f"\\nLoading and concatenating CSV files...")
dataframes = []
for csv_file in csv_files:
    print(f"  Loading {csv_file.name}...")
    df_temp = pd.read_csv(csv_file)
    print(f"    Rows: {len(df_temp)}, Columns: {len(df_temp.columns)}")
    dataframes.append(df_temp)

# Concatenate all dataframes
df = pd.concat(dataframes, ignore_index=True)
print(f"\\nTotal combined rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")

# Drop unnecessary columns
df = df.drop(columns=["Flow ID", "Src IP", "Dst IP", "Timestamp"], errors="ignore")
print(f"After dropping columns: {len(df.columns)} columns remaining")


# Extract and simplify label information
label_col = "label"

# Group similar attack types
# Apply simplification
df['label_simplified'] = df[label_col].apply(simplify_label)

# Group classes with less than 200 rows into "OTHERS"
label_counts = df['label_simplified'].value_counts()
min_rows = 200

# Find labels with less than min_rows
small_labels = label_counts[label_counts < min_rows].index.tolist()

if len(small_labels) > 0:
    print(f"Grouping {len(small_labels)} classes with < {min_rows} rows into 'OTHERS':")
    for small_label in small_labels:
        count = label_counts[small_label]
        print(f"  {small_label}: {count} rows -> OTHERS")
    
    # Replace small labels with "OTHERS"
    df.loc[df['label_simplified'].isin(small_labels), 'label_simplified'] = 'OTHERS'

# Create mapping to numeric
unique_labels = sorted(df['label_simplified'].unique())
label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
label_names_short = {idx: label for label, idx in label_mapping.items()}

df['label_numeric'] = df['label_simplified'].map(label_mapping)
label_col = 'label_numeric'
num_classes = len(unique_labels)

print(f"Found {num_classes} simplified label types:")
for orig_label in sorted(df['label_simplified'].unique()):
    num = label_mapping[orig_label]
    print(f"  {orig_label} -> {num}")

label_mapping_dict = label_names_short

print(f"\\nNumber of classes: {num_classes}")
print(f"Label distribution:\\n{df[label_col].value_counts().sort_index()}")
# -----------------------------'''

NEW = '''# 1. Load Dataset & Extract Multi-Class Labels (5-class CIC2017)
# -----------------------------
from utils.label_config import (
    DEFAULT_TRAINING_CSV,
    TARGET_CLASSES,
    DROPPED_CLASSES,
    build_label_mapping,
    filter_dataframe_to_target_classes,
)

DATASETS_FOLDER = Path("datasets")
csv_path = DATASETS_FOLDER / DEFAULT_TRAINING_CSV
if not csv_path.exists():
    raise FileNotFoundError(f"Training CSV not found: {csv_path}")

print(f"Loading {csv_path.name} (single-dataset training, avoids 37-class merge)...")
df = pd.read_csv(csv_path)
print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")

df = df.drop(columns=["Flow ID", "Src IP", "Dst IP", "Timestamp"], errors="ignore")
print(f"After dropping ID columns: {len(df.columns)} features")

label_col = "label"
df["label_simplified"] = df[label_col].apply(simplify_label)

before = len(df)
df, n_dropped = filter_dataframe_to_target_classes(df, "label_simplified")
print(f"Dropped {n_dropped} rows outside TARGET_CLASSES {list(TARGET_CLASSES)}")
print(f"Excluded label families (low prior F1 / not in 5-class set): {list(DROPPED_CLASSES)}")

label_mapping, label_names_short = build_label_mapping(TARGET_CLASSES)
df["label_numeric"] = df["label_simplified"].map(label_mapping)
label_col = "label_numeric"
num_classes = len(TARGET_CLASSES)

print(f"\\nFixed {num_classes}-class mapping:")
for name, idx in label_mapping.items():
    print(f"  {name} -> {idx}")

label_mapping_dict = label_names_short
print(f"\\nLabel distribution:\\n{df[label_col].value_counts().sort_index()}")
# -----------------------------'''

# Also patch imports cell
IMPORT_OLD = "from utils.vfl_utils import (\n    simplify_label,"
IMPORT_NEW = "from utils.label_config import TARGET_CLASSES, DEFAULT_TRAINING_CSV\nfrom utils.vfl_utils import (\n    simplify_label,"

# Patch markdown min_rows note
MD_OLD = "- `min_rows = 200`: long-tail threshold; classes below this are merged to `OTHERS` to stabilize multi-class learning."
MD_NEW = "- **5-class policy** (`utils/label_config.py`): train on CIC2017 only; keep BENIGN, DDOS, DOS, PORTSCAN, FTPPATATOR; drop BOT/WEBATTACK/SSHPATATOR/OTHERS and other datasets."

changed = False
for cell in nb["cells"]:
    src = "".join(cell.get("source", [])) if isinstance(cell.get("source"), list) else cell.get("source", "")
    if OLD in src:
        cell["source"] = NEW.splitlines(keepends=True)
        changed = True
    if IMPORT_OLD in src and "label_config" not in src:
        cell["source"] = src.replace(IMPORT_OLD, IMPORT_NEW).splitlines(keepends=True)
        changed = True
    if MD_OLD in src:
        cell["source"] = src.replace(MD_OLD, MD_NEW).splitlines(keepends=True)
        changed = True

if changed:
    nb_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("Train.ipynb patched for 5-class training")
else:
    print("Train.ipynb: no changes (block not found)")
