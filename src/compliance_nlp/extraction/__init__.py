"""Multi-task obligation extraction (NER + deontic + cross-ref)."""

from compliance_nlp.extraction.multitask import MultiTaskTrainer
from compliance_nlp.extraction.ner import decode_bio_tags, entity_level_f1
from compliance_nlp.extraction.deontic import keyword_classify, softmax_to_label
from compliance_nlp.extraction.crossref import (
    CrossReference,
    extract_citation_spans,
    resolve_cross_references,
)

__all__ = [
    "MultiTaskTrainer",
    "decode_bio_tags",
    "entity_level_f1",
    "keyword_classify",
    "softmax_to_label",
    "CrossReference",
    "extract_citation_spans",
    "resolve_cross_references",
]
