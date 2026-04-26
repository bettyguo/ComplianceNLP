"""Tests for multi-task obligation extraction."""

import pytest
import torch
from compliance_nlp.models.legal_bert import MultiTaskLegalBERT, CRFLayer
from compliance_nlp.data.datasets import ENTITY_TYPES, DEONTIC_LABELS


class TestCRFLayer:
    def test_init(self):
        crf = CRFLayer(num_tags=10)
        assert crf.num_tags == 10
        assert crf.transitions.shape == (10, 10)

    def test_decode_shape(self):
        crf = CRFLayer(num_tags=5)
        emissions = torch.randn(2, 10, 5)
        mask = torch.ones(2, 10)
        paths = crf.decode(emissions, mask)
        assert len(paths) == 2
        assert len(paths[0]) == 10


class TestMultiTaskLegalBERT:
    @pytest.fixture
    def model(self):
        return MultiTaskLegalBERT(
            backbone="bert-base-uncased",
            num_entity_types=len(ENTITY_TYPES),
            num_deontic_classes=len(DEONTIC_LABELS),
            use_crf=False,
        )

    def test_model_init(self, model):
        assert model is not None
        assert model.lambda_ner == 0.4
        assert model.lambda_deontic == 0.3
        assert model.lambda_xref == 0.3

    def test_forward_shape(self, model):
        input_ids = torch.randint(0, 1000, (2, 32))
        attention_mask = torch.ones(2, 32)
        ner_labels = torch.zeros(2, 32, dtype=torch.long)
        deontic_label = torch.zeros(2, dtype=torch.long)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            ner_labels=ner_labels,
            deontic_label=deontic_label,
        )

        assert "loss" in outputs
        assert "ner_logits" in outputs
        assert "deontic_logits" in outputs
        assert outputs["ner_logits"].shape == (2, 32, len(ENTITY_TYPES))
        assert outputs["deontic_logits"].shape == (2, len(DEONTIC_LABELS))
