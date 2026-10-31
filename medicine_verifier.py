"""
Local medicine-name verification and correction.

Given a name extracted from a prescription (OCR or typed), find and rank the
database names that are closest by **Levenshtein distance**, and return the
official name to use:

  * the database name itself when the spelling already matches, or
  * a corrected name (generic preferred when a brand is only marginally closer).

Also considered: known INN <-> USAN equivalents (paracetamol -> acetaminophen)
and pronunciation (Metaphone) similarity, which help with OCR slips.

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

try:  # Levenshtein distance (optional, pure-Python fallback below)
    from rapidfuzz.distance import Levenshtein
except ImportError:  # pragma: no cover
    Levenshtein = None

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

# How much each signal contributes to a candidate's score
LEVENSHTEIN_WEIGHT = 0.65
FUZZY_RATIO_WEIGHT = 0.35


def normalize_name(name):
    """Lowercase, strip punctuation and dosage-form words for comparison."""
    if not name:
        return ""
    text = name.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    tokens = [t for t in text.split() if t and t not in STOP_TOKENS]
    return " ".join(tokens).strip()


def _levenshtein_distance(a, b):
    """Plain edit distance, used when rapidfuzz is unavailable."""
    if a == b:
        return 0
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(min(previous[j] + 1,          # deletion
                               current[j - 1] + 1,       # insertion
                               previous[j - 1] + (char_a != char_b)))
        previous = current
    return previous[-1]


def levenshtein_similarity(a, b):
    """Normalized Levenshtein similarity: 1.0 identical, 0.0 unrelated."""
    if not a or not b:
        return 0.0
    if Levenshtein is not None:
        return float(Levenshtein.normalized_similarity(a, b))
    return 1.0 - _levenshtein_distance(a, b) / max(len(a), len(b))


def match_score(query, candidate):
    """Blend of Levenshtein similarity and fuzzy ratio (robust to OCR slips)."""
    score = levenshtein_similarity(query, candidate)
    if fuzz is not None:
        return LEVENSHTEIN_WEIGHT * score + FUZZY_RATIO_WEIGHT * (fuzz.WRatio(query, candidate) / 100.0)
    return score


def _is_subsequence(short, long):
    """True when every character of `short` appears in order inside `long`."""
    chars = iter(long)
    return all(char in chars for char in short)


def _truncated_from(norm, candidate):
    """True when `norm` looks like `candidate` after OCR dropped characters."""
    if len(norm) < 6 or len(candidate) <= len(norm):
        return False
    if candidate.startswith(norm) or candidate.endswith(norm):
        return False  # a real suffix/prefix ("augmentin" vs "augmentin xr")
    if not candidate.startswith(norm[:3]):
        return False  # OCR rarely loses the first letters
    if len(norm) / len(candidate) < 0.55:
        return False
    return _is_subsequence(norm, candidate)


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

    MIN_SCORE = 0.80           # confident enough to use as the correct name
    MIN_CANDIDATE_SCORE = 0.68  # lowest score worth showing as a suggestion
    AUTO_CORRECT_SCORE = 0.90  # correct without asking the user
    ALIAS_MIN_SCORE = 0.90     # known-equivalent names are curated, keep it strict
    GENERIC_PREFERENCE_MARGIN = 0.05
    PREFILTER_LIMIT = 25
    MAX_CANDIDATES = 5

    def __init__(self, csv_path=None, max_candidates=MAX_CANDIDATES):
        self.csv_path = csv_path or DEFAULT_DB_PATH
        self.max_candidates = max_candidates
        self.entries = {}         # normalized name -> {"name": display, "type": kind}
        self.phonetic_index = {}  # phonetic key -> [normalized names]
        self.aliases = {}         # normalized INN name -> {"usan": normalized, "name": display}
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
        self._names = list(self.entries)

    def _load_aliases(self):
        """Load known INN <-> USAN equivalent names (curated, see data/README.md)."""
        if not os.path.exists(ALIAS_PATH):
            return
        with open(ALIAS_PATH, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                inn = normalize_name(row.get("inn"))
                usan = normalize_name(row.get("usan"))
                if inn and usan and usan in self.entries:
                    self.aliases[inn] = {"usan": usan, "name": self.entries[usan]["name"]}

    @property
    def size(self):
        return len(self.entries)

    def verify(self, raw_name):
        """Rank database names for one extracted string and pick the official one."""
        raw_name = (raw_name or "").strip()
        norm = normalize_name(raw_name)
        result = {
            "input": raw_name,
            "normalized": norm,
            "exact_match": None,
            "phonetic_match": False,
            "candidates": [],
            "match_type": "none",
            "score": 0.0,
            "confidence": "low",
            "needs_confirmation": True,
            "auto_corrected": False,
            "corrected_name": raw_name,
        }
        if len(norm) < 3:
            return result

        # 1) Exact match on the normalized name
        exact = self.entries.get(norm)
        if exact:
            result.update({
                "exact_match": {"name": exact["name"], "type": exact["type"]},
                "candidates": [{
                    "name": exact["name"], "type": exact["type"],
                    "match_type": "exact", "score": 1.0,
                }],
                "match_type": "exact",
                "score": 1.0,
                "confidence": "high",
                "needs_confirmation": False,
                "auto_corrected": True,
                "corrected_name": exact["name"],
            })
            return result

        candidates = {}

        # 2) Known equivalent name (e.g. paracetamol <-> acetaminophen)
        alias = self._alias_match(norm)
        if alias:
            inn, usan, score = alias
            candidates[usan] = {
                "match_type": "alias",
                "score": score,
                "display": inn.title(),
                "equivalent": self.entries[usan]["name"],
            }

        # 3) Pronunciation-similar: same Metaphone key
        key = _phonetic_key(norm)
        for other in self.phonetic_index.get(key, []):
            similarity = _similarity(norm, other)
            if similarity < 0.75:  # same sound but too different to be a real typo
                continue
            score = max(0.85, similarity)
            if other not in candidates or candidates[other]["score"] < score:
                candidates[other] = {"match_type": "phonetic", "score": score}

        # 4) Spelling-similar: Levenshtein distance over the whole database
        for other, score in self._levenshtein_matches(norm):
            if other not in candidates or candidates[other]["score"] < score:
                candidates[other] = {"match_type": "spelling", "score": score}

        out = []
        for other, meta in self._rank(candidates)[: self.max_candidates]:
            entry = self.entries[other]
            item = {
                "name": meta.get("display") or entry["name"],
                "type": entry["type"],
                "match_type": meta["match_type"],
                "score": round(float(meta["score"]), 3),
            }
            if meta.get("equivalent"):
                item["equivalent"] = meta["equivalent"]
            out.append(item)

        result["candidates"] = out
        result["phonetic_match"] = any(c["match_type"] == "phonetic" for c in out)
        if out:
            best = out[0]
            result["match_type"] = best["match_type"]
            result["score"] = best["score"]
            dropped = (best["match_type"] == "spelling"
                       and _truncated_from(norm, normalize_name(best["name"])))
            if best["score"] >= self.AUTO_CORRECT_SCORE or dropped:
                # `dropped`: every typed character appears in order in the
                # official name, so OCR only lost characters - safe to correct.
                result["confidence"] = "high" if best["score"] >= self.AUTO_CORRECT_SCORE else "medium"
                result["needs_confirmation"] = False
                result["auto_corrected"] = True
                result["corrected_name"] = best["name"]
            elif best["score"] >= self.MIN_SCORE:
                result["confidence"] = "medium" if best["score"] >= 0.85 else "low"
                result["corrected_name"] = best["name"]
        return result

    def _rank(self, candidates):
        """Rank candidates, preferring an official generic name when close."""
        ranked = sorted(candidates.items(), key=lambda kv: kv[1]["score"], reverse=True)
        if len(ranked) > 1 and self.entries[ranked[0][0]]["type"] == "brand":
            best_score = ranked[0][1]["score"]
            for index in range(1, len(ranked)):
                name, meta = ranked[index]
                if (self.entries[name]["type"] == "generic"
                        and best_score - meta["score"] <= self.GENERIC_PREFERENCE_MARGIN):
                    ranked.insert(0, ranked.pop(index))
                    break
        return ranked

    def _alias_match(self, norm):
        """Find a known equivalent name, exactly or with a small typo."""
        exact = self.aliases.get(norm)
        if exact:
            return norm, exact["usan"], 0.98
        best = None
        for inn, meta in self.aliases.items():
            score = match_score(norm, inn)
            if score >= self.ALIAS_MIN_SCORE and (best is None or score > best[2]):
                best = (inn, meta["usan"], score)
        return best

    @staticmethod
    def _spelling_score(norm, candidate):
        """Levenshtein blend, lifted when OCR only dropped characters."""
        score = match_score(norm, candidate)
        if _truncated_from(norm, candidate):
            # The typed name is a gappy prefix of the candidate: rank by how
            # much of the official name survived instead of by edit distance.
            score = max(score, 0.60 + 0.40 * len(norm) / len(candidate))
        return score

    def _levenshtein_matches(self, norm):
        """Spelling-similar names, ranked by Levenshtein (+ fuzzy) similarity."""
        if not self._names:
            return []
        if process is not None and Levenshtein is not None:
            # Cheap, C-optimized prefilter; the blend is then applied to the pool.
            pool = [name for name, _, _ in process.extract(
                norm, self._names, scorer=Levenshtein.normalized_similarity,
                limit=self.PREFILTER_LIMIT)]
        else:
            pool = self._names
        scored = [(name, self._spelling_score(norm, name)) for name in pool]
        scored = [item for item in scored if item[1] >= self.MIN_CANDIDATE_SCORE]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored[: self.max_candidates * 2]

    def _norm_set(self):
        return self.entries
