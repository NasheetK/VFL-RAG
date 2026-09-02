#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.pipeline_paths import ensure_pipeline_dirs
ensure_pipeline_dirs()
print("Created output/inference|plans|scores class folders")
