"""Pure text-cleaning functions: HTML -> text, quote / signature / footer stripping.

Everything here is deterministic and dependency-light (BeautifulSoup only) so it can be
unit-tested exhaustively. The goal is to minimise tokens sent to the LLM *and* to reduce
prompt-injection surface (quoted history, hidden elements and boilerplate are removed).
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup

_ZERO_WIDTH_RE = re.compile("[\u200b-\u200f\u2060\ufeff\u034f\u00ad]")
_HIDDEN_STYLE_RE = re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden|max-height\s*:\s*0|font-size\s*:\s*0", re.I)

_DROP_TAGS = ["script", "style", "head", "title", "meta", "noscript", "template", "svg"]
_QUOTE_SELECTOR = "div.gmail_quote, blockquote, div.gmail_extra, div.yahoo_quoted, #divRplyFwdMsg, #appendonsend"
_BLOCK_TAGS = [
    "p", "div", "tr", "table", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6",
    "section", "article", "header", "footer", "pre",
]  # fmt: skip

_QUOTE_BOUNDARIES: tuple[re.Pattern[str], ...] = (
    re.compile(r"^[ \t]*On\s.{5,250}?\swrote:[ \t]*$", re.I | re.M | re.S),
    re.compile(r"^[ \t]*Le\s.{5,250}?\sa écrit\s?:[ \t]*$", re.I | re.M | re.S),
    re.compile(r"^[ \t]*-{2,}\s*Original Message\s*-{2,}[ \t]*$", re.I | re.M),
    re.compile(r"^[ \t]*_{10,}[ \t]*\n[ \t]*From:", re.I | re.M),
    re.compile(r"^[ \t]*From:[^\n]+\n[ \t]*(?:Sent|Date):[^\n]+\n", re.I | re.M),
)

_MOBILE_SIG_RE = re.compile(
    r"^\s*(sent from my |sent from outlook|get outlook for |sent via |sent using |sent from mail for )", re.I
)
_CLOSING_RE = re.compile(
    r"^(best regards|kind regards|warm regards|regards|best wishes|best|thanks|thank you|many thanks|"
    r"thanks in advance|cheers|sincerely|yours sincerely|yours truly|with thanks|cordially),?!?$",
    re.I,
)
_FOOTER_RE = re.compile(
    r"^\s*(this (e-?mail|message)[^\n]{0,80}(confidential|privileged|intended (solely )?for)"
    r"|confidentiality notice|disclaimer:|you are receiving this (e-?mail|message) because"
    r"|to unsubscribe|unsubscribe\b|view (this email )?in (your )?browser|manage (your )?(email )?preferences"
    r"|©\s?\d{4})",
    re.I | re.M,
)


def normalize_whitespace(text: str) -> str:
    """Normalise newlines, strip zero-width characters and collapse blank runs."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    text = _ZERO_WIDTH_RE.sub("", text)
    lines = [re.sub(r"[ \t]{2,}", " ", line).rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def html_to_text(html: str) -> str:
    """Convert an HTML email to readable plain text.

    Removes scripts/styles, hidden pre-header text, quoted reply blocks and tracking images;
    preserves paragraph / list structure.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(_DROP_TAGS):
        tag.decompose()
    for tag in soup.select(_QUOTE_SELECTOR):
        if not tag.decomposed:
            tag.decompose()
    for tag in soup.find_all(style=_HIDDEN_STYLE_RE):
        if not tag.decomposed:
            tag.decompose()
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for li in soup.find_all("li"):
        li.insert(0, "- ")
        li.insert_after("\n")
    for cell in soup.find_all(["td", "th"]):
        cell.insert_after(" ")
    for block in soup.find_all(_BLOCK_TAGS):
        block.insert_before("\n")
        block.insert_after("\n")
    return normalize_whitespace(soup.get_text())


def strip_quoted_replies(text: str) -> str:
    """Remove ``> quoted`` lines and everything after an "On ... wrote:" style boundary."""
    cut: int | None = None
    for pattern in _QUOTE_BOUNDARIES:
        for match in pattern.finditer(text):
            if match.group(0).count("\n") > 3:  # guard against runaway DOTALL matches
                continue
            if match.start() > 0 and text[: match.start()].strip():
                cut = match.start() if cut is None else min(cut, match.start())
            break
    if cut is not None:
        text = text[:cut]
    text = re.sub(r"^[ \t]*>.*$", "", text, flags=re.M)
    return text


def strip_signature(text: str) -> str:
    """Remove RFC-3676 ``-- `` signatures, mobile footers and short sign-off blocks."""
    lines = text.split("\n")

    for i, line in enumerate(lines):
        if i > 0 and line.rstrip() == "--":
            lines = lines[:i]
            break

    for i, line in enumerate(lines):
        if i > 0 and _MOBILE_SIG_RE.match(line):
            lines = lines[:i]
            break

    nonempty = [i for i, ln in enumerate(lines) if ln.strip()]
    window = nonempty[-12:]
    for idx in window:
        if _CLOSING_RE.match(lines[idx].strip()):
            tail = [ln for ln in lines[idx + 1 :] if ln.strip()]
            if len(tail) <= 8 and all(len(ln) <= 80 for ln in tail):
                lines = lines[: idx + 1]
            break
    return "\n".join(lines)


def strip_footer_boilerplate(text: str) -> str:
    """Cut legal disclaimers / unsubscribe footers that appear in the latter part of a message."""
    match = _FOOTER_RE.search(text)
    if match and match.start() >= len(text) * 0.4:
        return text[: match.start()]
    return text


def truncate(text: str, max_chars: int) -> str:
    """Keep head (80%) and tail (20%) of an over-long message."""
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.8)
    tail = max_chars - head
    omitted = len(text) - max_chars
    return f"{text[:head].rstrip()}\n\n[... {omitted} characters omitted ...]\n\n{text[-tail:].lstrip()}"


def clean_body(text: str, *, is_html: bool, max_chars: int = 4000) -> str:
    """Full cleaning pipeline.

    Falls back to the less-aggressively cleaned text if stripping would leave nothing
    (e.g. a message that *starts* with a forwarded header).
    """
    base = html_to_text(text) if is_html else normalize_whitespace(text)
    cleaned = normalize_whitespace(strip_footer_boilerplate(strip_signature(strip_quoted_replies(base))))
    if not cleaned:
        cleaned = base
    return truncate(cleaned, max_chars)
