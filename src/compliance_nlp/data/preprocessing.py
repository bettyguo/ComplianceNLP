"""Regulatory text preprocessing utilities.

Handles format-specific parsing for SEC EDGAR XML, EUR-Lex HTML,
and BIS PDF documents, plus common regulatory text normalization.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

log = logging.getLogger(__name__)


@dataclass
class RegulatoryChunk:
    """A chunk of regulatory text with metadata."""

    text: str
    provision_id: str
    framework: str  # SEC, MiFID II, Basel III
    source_url: str = ""
    effective_date: str = ""
    chunk_id: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Text Normalization
# ─────────────────────────────────────────────────────────────────────────────

# Regex patterns for cross-reference detection
XREF_PATTERNS = {
    "sec": [
        r"(?:Section|§)\s*\d+[\w.]*(?:\([a-zA-Z0-9]+\))*",
        r"(?:Rule|Regulation)\s+\d+[a-zA-Z]*-\d+",
        r"(?:Item)\s+\d+(?:\([a-zA-Z0-9]+\))*",
        r"17\s*CFR\s*§?\s*\d+\.\d+",
    ],
    "mifid": [
        r"Art(?:icle)?\.?\s*\d+(?:\(\d+\))*",
        r"Directive\s+\d{4}/\d+/EU",
        r"Regulation\s+\(EU\)\s+\d{4}/\d+",
        r"Delegated\s+Regulation\s+\(EU\)\s+\d{4}/\d+",
    ],
    "basel": [
        r"(?:paragraph|para\.?|¶)\s*\d+(?:--\d+)?",
        r"BCBS\s+d\d+",
        r"CRR\s+Art(?:icle)?\.?\s*\d+",
        r"CET1|AT1|Tier\s*[12]",
    ],
}


def normalize_regulatory_text(text: str) -> str:
    """Normalize regulatory text for consistent processing.

    Args:
        text: Raw regulatory text.

    Returns:
        Normalized text with standardized whitespace and encoding.
    """
    # Normalize Unicode characters
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "--").replace("\u2014", "---")
    text = text.replace("\xa0", " ")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Standardize section references
    text = re.sub(r"Section\s+", "Section ", text)
    text = re.sub(r"Article\s+", "Article ", text)

    return text


def extract_cross_references(text: str, framework: str) -> list[str]:
    """Extract cross-reference mentions from regulatory text.

    Args:
        text: Regulatory text to parse.
        framework: One of 'sec', 'mifid', 'basel'.

    Returns:
        List of extracted cross-reference strings.
    """
    framework_key = framework.lower().replace(" ", "").replace("_", "")
    if "mifid" in framework_key or "eu" in framework_key:
        framework_key = "mifid"
    elif "basel" in framework_key:
        framework_key = "basel"
    else:
        framework_key = "sec"

    patterns = XREF_PATTERNS.get(framework_key, [])
    references = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        references.extend(matches)

    return list(set(references))


def chunk_regulatory_document(
    text: str,
    provision_id: str,
    framework: str,
    max_chunk_size: int = 512,
    overlap: int = 50,
) -> list[RegulatoryChunk]:
    """Split a regulatory document into overlapping chunks.

    Args:
        text: Full document text.
        provision_id: Identifier for the source provision.
        framework: Regulatory framework identifier.
        max_chunk_size: Maximum tokens per chunk (approximate by words).
        overlap: Number of overlapping words between chunks.

    Returns:
        List of RegulatoryChunk objects.
    """
    words = text.split()
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(words):
        end = min(start + max_chunk_size, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append(
            RegulatoryChunk(
                text=normalize_regulatory_text(chunk_text),
                provision_id=f"{provision_id}_chunk{chunk_id}",
                framework=framework,
                chunk_id=chunk_id,
            )
        )
        chunk_id += 1
        start = end - overlap if end < len(words) else end

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Format-Specific Parsers
# ─────────────────────────────────────────────────────────────────────────────


def parse_sec_edgar_xml(filepath: str | Path) -> Iterator[RegulatoryChunk]:
    """Parse SEC EDGAR XML filing into regulatory chunks.

    Args:
        filepath: Path to SEC EDGAR XML file.

    Yields:
        RegulatoryChunk objects for each provision/section.
    """
    import xml.etree.ElementTree as ET

    filepath = Path(filepath)
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
    except ET.ParseError as e:
        log.error(f"Failed to parse SEC XML file {filepath}: {e}")
        return

    # Extract sections from EDGAR XML structure
    ns = {"sec": "http://www.sec.gov/edgar/common"}
    for section in root.iter():
        if section.text and len(section.text.strip()) > 50:
            text = normalize_regulatory_text(section.text)
            tag_name = section.tag.split("}")[-1] if "}" in section.tag else section.tag
            provision_id = f"SEC_{tag_name}_{section.get('id', 'unknown')}"
            for chunk in chunk_regulatory_document(text, provision_id, "SEC"):
                yield chunk


def parse_eurlex_html(filepath: str | Path) -> Iterator[RegulatoryChunk]:
    """Parse EUR-Lex HTML document into regulatory chunks.

    Args:
        filepath: Path to EUR-Lex HTML file.

    Yields:
        RegulatoryChunk objects for each article/provision.
    """
    from html.parser import HTMLParser

    filepath = Path(filepath)
    text = filepath.read_text(encoding="utf-8")

    # Simple extraction: split by article markers
    articles = re.split(r"(?i)(?=Article\s+\d+)", text)
    for i, article_text in enumerate(articles):
        if len(article_text.strip()) < 50:
            continue
        # Extract article number
        match = re.match(r"(?i)Article\s+(\d+)", article_text)
        article_num = match.group(1) if match else str(i)
        clean_text = re.sub(r"<[^>]+>", " ", article_text)
        clean_text = normalize_regulatory_text(clean_text)
        provision_id = f"MiFID_Art_{article_num}"
        for chunk in chunk_regulatory_document(clean_text, provision_id, "MiFID II"):
            yield chunk


def parse_bis_pdf(filepath: str | Path) -> Iterator[RegulatoryChunk]:
    """Parse BIS PDF document into regulatory chunks.

    Args:
        filepath: Path to BIS/Basel Committee PDF file.

    Yields:
        RegulatoryChunk objects for each paragraph/provision.
    """
    filepath = Path(filepath)
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(str(filepath))
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()
    except ImportError:
        log.warning("PyMuPDF not installed; falling back to basic text extraction")
        full_text = filepath.read_text(encoding="utf-8", errors="replace")

    # Split by paragraph numbers
    paragraphs = re.split(r"(?=\d+\.\s)", full_text)
    for i, para_text in enumerate(paragraphs):
        if len(para_text.strip()) < 50:
            continue
        match = re.match(r"(\d+)\.\s", para_text)
        para_num = match.group(1) if match else str(i)
        clean_text = normalize_regulatory_text(para_text)
        provision_id = f"Basel_para_{para_num}"
        for chunk in chunk_regulatory_document(clean_text, provision_id, "Basel III"):
            yield chunk
