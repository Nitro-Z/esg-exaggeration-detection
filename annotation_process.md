# Annotation Process Description

## Overview

This document describes the annotation process used to construct the 300-sentence gold standard dataset for exaggeration detection in English-language ESG reports. The dataset was produced through independent human annotation by two annotators and serves as the evaluation benchmark for this study.

---

## Annotators

Two annotators participated in the task:

- **Annotator 1**: The primary researcher, a graduate student with domain familiarity in ESG disclosure and sustainability reporting.
- **Annotator 2**: An independent annotator with no prior involvement in the study, recruited to provide an external perspective.

Both annotators are native or near-native readers of English. Neither annotator was provided with a formal ruleset prior to the task. Labeling decisions were based on each annotator's individual linguistic intuition regarding what constitutes exaggeratory language in the context of corporate sustainability disclosure.

---

## Annotation Task

Each annotator was presented with 300 sentences drawn from the annotation subset (`annotation_sample/`), sampled from 50 ESG reports not included in the training or test corpus. Annotators worked independently with no communication during the task.

Each sentence was assigned a binary label:

| Label | Meaning |
|-------|---------|
| `1`   | Exaggeration present — the sentence contains language that overstates, inflates, or makes implausible claims about the company's sustainability performance or commitments |
| `0`   | No exaggeration — the sentence makes a factual, measured, or qualified claim without apparent inflation |

---

## Labeling Approach

No explicit annotation guidelines or decision trees were provided prior to the task. Both annotators relied on their personal judgment of whether a sentence's language was disproportionate to what could reasonably be substantiated — for example, absolute claims of environmental perfection, superlative assertions without qualification, or vague but sweeping commitments presented as established fact.

This intuition-based approach was intentional: one objective of the annotation exercise was to assess whether exaggeration in ESG language is recognizable to independent human readers without domain-specific training, thereby testing the face validity of the construct.

---

## Adjudication

After independent annotation, disagreements between the two annotators were resolved through discussion between the primary researcher and Annotator 2. In cases where consensus could not be reached, a **conservative rule** was applied: the sentence was assigned `label = 0` (no exaggeration), reflecting a preference for false negatives over false positives in the gold standard.

---

## Inter-Annotator Agreement

Agreement was measured on a shared subset of 100 sentences annotated by both annotators. Cohen's kappa (κ) was computed as the primary agreement metric, yielding **κ = 0.878**, indicating near-perfect agreement and suggesting that exaggeration in ESG disclosure is a sufficiently salient linguistic phenomenon to be reliably identified through intuitive judgment alone.

---

## Dataset Composition

| Attribute | Value |
|-----------|-------|
| Total sentences | 300 |
| Positive instances (label = 1) | 15 |
| Negative instances (label = 0) | 285 |
| Source reports | 50 ESG reports |
| Shared subset for IAA | 100 sentences |
| Inter-annotator agreement (κ) | 0.878 |

---

## Notes on Construct Validity

The absence of a formal annotation ruleset is acknowledged as a limitation of this study. Future work may benefit from developing explicit operationalization criteria for exaggeration in ESG disclosure, potentially drawing on the lexicon developed in this study as a basis for more structured annotation protocols. The high inter-annotator agreement observed here, however, provides preliminary evidence that the construct is interpretable without such formalization.
