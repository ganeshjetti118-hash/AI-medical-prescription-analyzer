"""
Local medicine-name verification.

Given a name extracted from a prescription (OCR or typed), find:
  * an exact match in the local medicine database, or
  * candidates that share the same spelling (fuzzy) or pronunciation (phonetic).

Everything runs locally; no name or image leaves the machine.

Data: data/medicine_names.csv, built from the FDA Orange Book by
scripts/build_medicine_db.py (see data/README.md for provenance and licence).
"""
import csv
import os
import re
from difflib import SequenceMatcher

try:  # fast fuzzy matching (optional)
    from rapidfuzz import fuzz, process
except ImportError:  # pragma: no cover - fallback keeps the feature working
    fuzz = None
    process = None

try:  # phonetic matching (optional)
    import jellyfish
except ImportError:  # pragma: no cover
    jellyfish = None

# Dosage forms and prescription noise that are not part of a medicine name
STOP_TOKENS = {
    "tab", "tabs", "tablet", "tablets", "cap", "caps", "capsule", "capsules",
    "syp", "syrup", "inj", "injection", "susp", "suspension", "drop", "drops",
    "oint", "ointment", "cream", "gel", "lotion", "spray", "inhaler", "respules",
    "rx", "sig", "od", "bd", "tds", "qid", "hs", "prn", "sos", "stat",
    "mg", "ml", "mcg", "gm", "g",
    "iu", "units", "unit", "nos", "no",
}

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "data", "medicine_names.csv")
ALIAS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "data", "inn_aliases.csv")


def normalize_name(name):
    """Lowercase, strip punctuation and dosage-form words for comparison."""
    if not name:
        return ""
    text = name.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    tokens = [t for t in text.split() if t and t not in STOP_TOKENS]
    return " ".join(tokens).strip()


def _phonetic_key(name):
    if jellyfish is None:
        return None
    key = jellyfish.metaphone(name)
    return key or None


def _similarity(a, b):
    # Plain character ratio avoids partial-match inflation on short names.
    if fuzz is not None:
        return fuzz.ratio(a, b) / 100.0
    return SequenceMatcher(None, a, b).ratio()


class MedicineVerifier:
    """Exact + spelling-similar + pronunciation-similar lookup in a local DB."""

    MIN_SCORE = 0.82
    MAX_CANDIDATES = 5

    def __init__(self, csv_path=None, max_candidates=MAX_CANDIDATES):
        self.csv_path = csv_path or DEFAULT_DB_PATH
        self.max_candidates = max_candidates
        self.entries = {}       # normalized name -> {"name": display, "type": kind}
        self.phonetic_index = {}  # phonetic key -> [normalized names]
        self.aliases = {}       # normalized INN name -> normalized USAN name
        self._load()
        self._load_aliases()

    def _load(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(
                f"Medicine database not found at {self.csv_path}. "
                "Run: python scripts/build_medicine_db.py"
            )
        with open(self.csv_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                display = (row.get("name") or "").strip()
                norm = normalize_name(display)
                if len(norm) < 3 or norm in self.entries:
                    continue
                self.entries[norm] = {"name": display, "type": row.get("type") or "generic"}
                key = _phonetic_key(norm)
                if key:
                    self.phonetic_index.setdefault(key, []).append(norm)

    def _load_aliases(self):
        """Load known INN <-> USAN equivalent names (curated, see data/README.md)."""
        if not os.path.exists(ALIAS_PATH):
            return
        with open(ALIAS_PATH, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                inn = normalize_name(row.get("inn"))
                usan = normalize_name(row.get("usan"))
                if inn and usan and usan in self.entries:
                    self.aliases[inn] = usan

    @property
    def size(self):
        return len(self.entries)

    def verify(self, raw_name):
        """Return the match status for one extracted medicine name."""
        norm = normalize_name(raw_name)
        result = {
            "input": raw_name,
            "normalized": norm,
            "exact_match": None,
            "phonetic_match": False,
            "candidates": [],
            "confidence": "low",
            "needs_confirmation": True,
        }
        if len(norm) < 3:
            return result

        # 1) Exact match on the normalized name
        exact = self.entries.get(norm)
        if exact:
            result["exact_match"] = {"name": exact["name"], "type": exact["type"]}
            result["candidates"] = [{
                "name": exact["name"], "type": exact["type"],
                "match_type": "exact", "score": 1.0,
            }]
            result["confidence"] = "high"
            result["needs_confirmation"] = False
            return result

        candidates = {}

        # 2) Known equivalent name (e.g. paracetamol -> acetaminophen)
        canonical = self.aliases.get(norm)
        if canonical:
            candidates[canonical] = {"match_type": "alias", "score": 0.98}

        # 3) Pronunciation-similar: same Metaphone key
        key = _phonetic_key(norm)
        for other in self.phonetic_index.get(key, []):
            similarity = _similarity(norm, other)
            if similarity < 0.75:  # same sound but too different to be a real typo
                continue
            score = max(0.85, similarity)
            if other not in candidates or candidates[other]["score"] < score:
                candidates[other] = {"match_type": "phonetic", "score": score}

        # 4) Spelling-similar: fuzzy match over the whole database
        for other, score in self._fuzzy_matches(norm):
            if other not in candidates or candidates[other]["score"] < score:
                candidates[other] = {"match_type": "spelling", "score": score}

        ranked = sorted(candidates.items(), key=lambda kv: kv[1]["score"], reverse=True)
        out = []
        for other, meta in ranked[: self.max_candidates]:
            entry = self.entries[other]
            out.append({
                "name": entry["name"],
                "type": entry["type"],
                "match_type": meta["match_type"],
                "score": round(float(meta["score"]), 3),
            })

        result["candidates"] = out
        result["phonetic_match"] = any(c["match_type"] == "phonetic" for c in out)
        if out:
            best = out[0]["score"]
            result["confidence"] = "medium" if best >= 0.9 else "low"
        return result

    def _norm_set(self):
        return self.entries

    def _fuzzy_matches(self, norm):
        names = list(self.entries.keys())
        if not names:
            return []
        if process is not None:
            matches = process.extract(norm, names, scorer=fuzz.ratio,
                                      limit=self.max_candidates * 2,
                                      score_cutoff=self.MIN_SCORE * 100)
            return [(name, score / 100.0) for name, score, _ in matches]
        scored = []
        for name in names:
            score = SequenceMatcher(None, norm, name).ratio()
            if score >= self.MIN_SCORE:
                scored.append((name, score))
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored[: self.max_candidates * 2]
