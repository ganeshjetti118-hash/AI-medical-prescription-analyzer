"""
Build the local medicine-name database from the FDA Orange Book.

Source: FDA Orange Book data files (public domain, U.S. Government work).
  https://www.fda.gov/drug/drugs-fda-orange-book
  Direct download: https://www.fda.gov/media/76860/download (products.txt)

Output: data/medicine_names.csv  (columns: name,type,source)

Refresh:
    docker compose -f docker-compose.base44.yml exec -T web python scripts/build_medicine_db.py

The generated CSV is committed so the app has no network dependency at runtime.
"""
import csv
import io
import os
import re
import sys
import zipfile

import requests

ORANGE_BOOK_URL = "https://www.fda.gov/media/76860/download"
OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "medicine_names.csv")

# Salt / hydrate suffixes: also index the base molecule name (e.g. "metformin hydrochloride" -> "metformin")
SALT_SUFFIXES = [
    "hydrochloride", "hcl", "hydrobromide", "sodium", "potassium", "calcium",
    "sulfate", "sulphate", "tartrate", "citrate", "maleate", "mesylate",
    "besylate", "fumarate", "succinate", "phosphate", "acetate", "dihydrate",
    "monohydrate", "hydrate", "bromide", "chloride", "nitrate", "oxalate",
    "lactate", "gluconate", "carbonate", "stearate", "palmitate", "benzoate",
    "malate", "tosylate", "bitartrate", "hemifumarate", "xinafoate",
]

INVALID_WORDS = {"and", "with", "in", "of", "the", "for"}


def clean(name):
    """Normalize a raw Orange Book name into a display name, or return None."""
    if not name:
        return None
    name = name.strip().strip('"').strip()
    if not name:
        return None
    # Orange Book wraps multi-word names in quotes in some releases
    name = name.replace('"', "").strip()
    if not re.search(r"[A-Za-z]{3,}", name):
        return None
    if any(ch.isdigit() for ch in name):
        return None
    lowered = name.lower()
    if lowered in INVALID_WORDS or len(lowered) < 4:
        return None
    # Keep readable display casing, collapse whitespace
    return " ".join(name.split())


def add_entry(entries, name, kind):
    name = clean(name)
    if not name:
        return
    key = name.lower()
    existing = entries.get(key)
    # Prefer generic over brand when the same spelling appears twice
    if existing is None or (existing[1] == "brand" and kind == "generic"):
        entries[key] = (name, kind)


def base_name(ingredient):
    """Strip trailing salt/hydrate words to add the parent molecule name."""
    words = ingredient.lower().split()
    stripped = list(words)
    while len(stripped) > 1 and stripped[-1] in SALT_SUFFIXES:
        stripped.pop()
    if stripped and stripped != words and len(" ".join(stripped)) >= 4:
        return " ".join(stripped)
    return None


def download_products():
    # FDA blocks default programmatic agents; a normal browser UA is required.
    headers = {"User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")}
    resp = requests.get(ORANGE_BOOK_URL, timeout=120, headers=headers)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        target = next(n for n in zf.namelist() if n.lower().endswith("products.txt"))
        return zf.read(target).decode("latin-1")


def main():
    print("Downloading FDA Orange Book data...")
    raw = download_products()
    entries = {}

    # The products file has shipped tab- and tilde-delimited; detect it.
    delimiter = "~" if "~" in raw.splitlines()[0] else "\t"
    reader = csv.reader(io.StringIO(raw), delimiter=delimiter)
    header = next(reader)
    idx = {col.strip().lower(): i for i, col in enumerate(header)}
    i_ingredient = idx.get("ingredient")
    i_trade = idx.get("trade_name")
    if i_ingredient is None:
        sys.exit(f"Unexpected Orange Book header: {header}")

    rows = 0
    for row in reader:
        if len(row) <= max(i_ingredient, i_trade or 0):
            continue
        rows += 1
        ingredient_field = row[i_ingredient]
        for part in ingredient_field.split(";"):
            ingredient = clean(part)
            if ingredient:
                add_entry(entries, ingredient, "generic")
                base = base_name(ingredient)
                if base:
                    add_entry(entries, base, "generic")
        if i_trade is not None:
            add_entry(entries, row[i_trade], "brand")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["name", "type", "source"])
        for name, kind in sorted(entries.values(), key=lambda e: e[0].lower()):
            writer.writerow([name, kind, "FDA Orange Book"])

    generics = sum(1 for _, k in entries.values() if k == "generic")
    brands = len(entries) - generics
    print(f"Parsed {rows} products -> {len(entries)} names "
          f"({generics} generic, {brands} brand) -> {OUT_PATH}")


if __name__ == "__main__":
    main()
