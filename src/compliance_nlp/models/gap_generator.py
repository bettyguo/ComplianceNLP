"""LLaMA-3-based compliance gap analysis generator.

Generates structured gap reports by comparing extracted obligations
against internal policies, with severity-aware scoring and
source-grounded justifications.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)


@dataclass
class GapReport:
    """Structured compliance gap analysis output."""

    classification: str  # Compliant, Partial Gap, Full Gap
    severity: str  # Minor, Moderate, Major, Critical
    alignment_score: float
    matched_policy: str
    gap_description: str
    recommended_action: str
    grounding_sentences: int
    grounding_verified: int
    grounding_confidence: float


GAP_ANALYSIS_PROMPT = """You are a regulatory compliance expert analyzing whether an institution's internal policies adequately cover a specific regulatory obligation.

## Regulatory Obligation
Entity: {entity}
Action: {action}
Modality: {modality}
Condition: {condition}
Source: {source_provision}
Cross-references: {cross_references}

## Retrieved Context
{retrieved_context}

## Internal Policy
Section: {policy_section}
Text: {policy_text}

## Task
Analyze whether the internal policy adequately addresses the regulatory obligation. Classify as:
- **Compliant**: Policy fully covers the obligation
- **Partial Gap**: Policy partially covers but misses key aspects
- **Full Gap**: Policy does not address the obligation

Provide your analysis in the following format:
Classification: [Compliant/Partial Gap/Full Gap]
Severity: [N/A/Minor/Moderate/Major/Critical]
Gap Description: [Detailed description of any gaps found]
Recommended Action: [Specific remediation steps if gaps exist]
"""


class GapAnalysisGenerator:
    """LLaMA-3-based generator for compliance gap analysis.

    Generates structured gap reports conditioned on extracted obligations,
    retrieved regulatory context, and internal policy text.
    """

    def __init__(
        self,
        model_path: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        device: str = "auto",
        max_length: int = 512,
        temperature: float = 0.1,
        torch_dtype: torch.dtype = torch.float16,
    ):
        self.max_length = max_length
        self.temperature = temperature

        log.info(f"Loading generator model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        log.info("Generator model loaded")

    def generate_gap_report(
        self,
        entity: str,
        action: str,
        modality: str,
        condition: str,
        source_provision: str,
        cross_references: list[str],
        retrieved_context: str,
        policy_section: str,
        policy_text: str,
    ) -> GapReport:
        """Generate a compliance gap analysis report.

        Args:
            entity: Regulated entity type.
            action: Required action from the obligation.
            modality: Deontic modality (Obligation, Permission, etc.).
            condition: Conditions under which the obligation applies.
            source_provision: Source regulatory provision reference.
            cross_references: List of cross-referenced provisions.
            retrieved_context: KG-augmented retrieved passages.
            policy_section: Internal policy section identifier.
            policy_text: Internal policy clause text.

        Returns:
            Structured GapReport with classification, severity, and details.
        """
        prompt = GAP_ANALYSIS_PROMPT.format(
            entity=entity,
            action=action,
            modality=modality,
            condition=condition,
            source_provision=source_provision,
            cross_references=", ".join(cross_references) if cross_references else "None",
            retrieved_context=retrieved_context,
            policy_section=policy_section,
            policy_text=policy_text,
        )

        # Format as chat messages for instruct model
        messages = [
            {"role": "system", "content": "You are a regulatory compliance expert."},
            {"role": "user", "content": prompt},
        ]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(
            formatted, return_tensors="pt", truncation=True, max_length=4096
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_length,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        return self._parse_gap_report(generated_text)

    def _parse_gap_report(self, text: str) -> GapReport:
        """Parse generated text into structured GapReport.

        Args:
            text: Raw generated text from the model.

        Returns:
            Parsed GapReport object.
        """
        lines = text.strip().split("\n")
        fields = {}
        current_key = ""
        for line in lines:
            for key in ["Classification", "Severity", "Gap Description", "Recommended Action"]:
                if line.strip().startswith(key + ":"):
                    current_key = key
                    fields[key] = line.split(":", 1)[1].strip()
                    break
            else:
                if current_key and line.strip():
                    fields[current_key] = fields.get(current_key, "") + " " + line.strip()

        classification = fields.get("Classification", "Full Gap").strip()
        if classification not in ["Compliant", "Partial Gap", "Full Gap"]:
            classification = "Full Gap"  # Conservative default

        severity = fields.get("Severity", "Major").strip()
        if severity not in ["N/A", "Minor", "Moderate", "Major", "Critical"]:
            severity = "Major"

        return GapReport(
            classification=classification,
            severity=severity if classification != "Compliant" else "N/A",
            alignment_score=0.0,  # Populated by alignment module
            matched_policy="",
            gap_description=fields.get("Gap Description", ""),
            recommended_action=fields.get("Recommended Action", ""),
            grounding_sentences=0,
            grounding_verified=0,
            grounding_confidence=0.0,
        )

    def batch_generate(
        self, batch: list[dict], max_concurrent: int = 4
    ) -> list[GapReport]:
        """Generate gap reports for a batch of obligation-policy pairs.

        Args:
            batch: List of dicts with obligation and policy fields.
            max_concurrent: Maximum concurrent generations.

        Returns:
            List of GapReport objects.
        """
        results = []
        for item in batch:
            report = self.generate_gap_report(**item)
            results.append(report)
        return results
