"""
Manual end-to-end check for the GEO fetch mode.

Pulls MINiML XML directly from NCBI for a hand-picked mix of GSM and GSE
accession IDs and runs factor extraction (cheapest LLM mode) so you can
eyeball the results.

Run with:
    ./venv/bin/python tests/scratch/manual_fetch_check.py
"""
import json
import os
import sys
from pathlib import Path

# Load .env so OPENAI_API_KEY (or whichever provider) is picked up.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from CistromeMetaX import meta_extract_factors
from CistromeMetaX.geo_fetch import expand_gse_to_gsms

# 10 GSMs + 1 GSE. All human ChIP-seq, diverse non-histone transcription factors.
# The GSE will be auto-expanded to its child GSMs.
GSM_IDS = [
    "GSM935373",    # GATA-2  (K562)
    "GSM935274",    # CEBPZ   (HepG2)
    "GSM803411",    # ZEB1    (GM12878)
    "GSM935317",    # c-Fos   (HeLa-S3)
    "GSM935616",    # BHLHE40 (K562)
    "GSM1010812",   # TAF1    (A549)
    "GSM1010895",   # TEAD4   (K562)
    "GSM782123",    # TCF7L2  (HCT-116)
    "GSM935496",    # TAL1    (K562)
    "GSM935401",    # p300    (K562)
]
GSE_IDS = ["GSE40398"]   # small GSE (~6 children) — auto-expands to child GSMs


def main():
    accessions = GSM_IDS + GSE_IDS

    # Preview how many GSMs we are about to run through the LLM so the user
    # can bail out before burning tokens if a GSE is huge.
    print(f"Input: {len(GSM_IDS)} GSMs + {len(GSE_IDS)} GSE")
    total_from_gses = 0
    for gse in GSE_IDS:
        children = expand_gse_to_gsms(gse, verbose=True)
        print(f"  {gse} -> {len(children)} child GSMs")
        total_from_gses += len(children)
    print(f"Estimated GSM extractions (pre-dedup): {len(GSM_IDS) + total_from_gses}")
    print("-" * 60)

    results = meta_extract_factors(accessions, model="openai:gpt-5", verbose=True)

    print("-" * 60)
    print(f"Returned {len(results)} result entries.")
    out_path = Path(__file__).parent / "manual_fetch_check.results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Full results written to: {out_path}")

    # Pretty-print first 3 entries to stdout for quick inspection.
    print("\nFirst 3 entries:")
    print(json.dumps(results[:3], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
