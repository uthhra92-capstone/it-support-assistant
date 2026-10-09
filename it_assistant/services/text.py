"""Small text utilities shared by search and duplicate detection."""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a an and are as at be but by can do does for from has have how i if in is it its
    me my not of on or our please should so that the their them then there this to up
    us was we what when where which who why will with would you your am been being did
    get got cannot cant dont im ive""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens with stopwords removed and light plural stemming."""
    tokens = []
    for tok in _TOKEN_RE.findall(text.lower()):
        if tok in STOPWORDS or len(tok) < 2:
            continue
        if len(tok) > 4 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        tokens.append(tok)
    return tokens


def jaccard_similarity(a: str, b: str) -> float:
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)
