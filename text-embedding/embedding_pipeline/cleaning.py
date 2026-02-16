from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Dict

# NOTE: Cleaning is optional. Default pipeline behavior is "no text modification".


def load_rules(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U00002600-\U000026FF"
    "]+",
    flags=re.UNICODE,
)

URL_RE = re.compile(r"(https?://\\S+|www\\.\\S+)", flags=re.IGNORECASE)
MENTION_RE = re.compile(r"@\\w+", flags=re.UNICODE)
HASHTAG_RE = re.compile(r"#\\w+", flags=re.UNICODE)
MULTI_WS_RE = re.compile(r"\\s+")


def clean_text(text: Any, rules: Dict[str, Any]) -> str:
    """
    Optional cleaning function (ported from the user's OpenAI embedding repo).
    Keep this disabled unless explicitly requested.
    """
    if text is None:
        return ""
    s = str(text)
    if rules.get("normalize_unicode_nfkc", True):
        s = unicodedata.normalize("NFKC", s)
    if rules.get("remove_control_chars", True):
        s = "".join(ch for ch in s if unicodedata.category(ch) not in {"Cc", "Cf"})
    if rules.get("remove_urls", True):
        s = URL_RE.sub(" ", s)
    if rules.get("remove_mentions", False):
        s = MENTION_RE.sub(" ", s)
    if rules.get("remove_hashtags", False):
        s = HASHTAG_RE.sub(" ", s)
    if rules.get("remove_emojis", False):
        s = EMOJI_RE.sub(" ", s)
    if rules.get("lowercase_latin", False):
        s = s.lower()
    if rules.get("collapse_whitespace", True):
        s = MULTI_WS_RE.sub(" ", s)
    if rules.get("strip", True):
        s = s.strip()
    max_len = int(rules.get("max_length", 0) or 0)
    if max_len > 0:
        s = s[:max_len]
    return s

