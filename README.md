# Detection of Exaggeration in English ESG Disclosures
## Evidence from Taiwanese Listed Firms via Machine Learning


**Author:** Pham Khac Nam  

---

## Overview

This repository contains the code, lexicon, and supplementary materials for a study on automated detection of exaggeratory language in English-language ESG reports published by TWSE- and TPEx-listed Taiwanese companies.

The methodology combines a domain-specific 281-term exaggeration lexicon (grounded in Appraisal Theory) with supervised machine learning classification (Logistic Regression, SVM, Random Forest x TF-IDF / Word2Vec), evaluated via a two-tier framework: internal evaluation on silver-labeled test data and external evaluation on a 300-sentence human-annotated gold standard (k=0.878).

**Key results:**
- Corpus: 982 ESG reports, 391 companies, ~1.38 million sentences
- Exaggeration ratio: 2.3% of sentences
- Best model: Random Forest + TF-IDF (F1=0.500, Precision=0.991, AUC=0.908)
- Gold standard evaluation: F1=0.381 vs. dictionary baseline F1=0.692

---

## Repository Structure

```
├── pipeline_classification.py        # Main ML pipeline (labeling, training, evaluation)
├── Draw_charts.py                    # Thesis figure generation
├── export_lexicon.py                 # Lexicon frequency export
├── shuffle_dataset.py                # Corpus partitioning (train/val/test/annotation)
├── PDF_to_TEXT.py                    # PDF-to-text conversion
├── Format_Dict.py                    # Lexicon formatting utilities
├── Dict_Performance.py               # Dictionary baseline evaluation
├── annotation.py                     # Annotation interface
├── annotation_tasks.py               # Annotation task management
├── export_vader.py                   # VADER lexicon filtering
├── check_frequency.py                # Term frequency checking
├── lexicon_full.csv                  # Finalized 281-term lexicon with corpus frequencies
├── All_companies_with_industry.xlsx  # Corpus company list (391 firms, TWSE classification)
├── annotation_process.md             # Annotation methodology description
└── .gitignore
```

---

## Supplementary Materials

The following resources are provided as supplementary material:

- **Appendix A** — Full exaggeration lexicon (281 terms, tier classification, corpus frequency): `lexicon_full.csv`
- **Appendix B** — Annotation process description: `annotation_process.md`
- **Appendix C** — Gold standard dataset (300 sentences, human-annotated): available on request
- **Appendix D** — Corpus company list: `All_companies_with_industry.xlsx`

> The full training/validation/test corpus (982 reports) is not included in this repository due to copyright restrictions on the original ESG report content. A sample of 50 reports used for annotation is provided in `Dataset/annotation_sample/` for reference and testing purposes.
>
> **To request access to the full corpus, please contact:** phamkhacnam2001@gmail.com

---

## Requirements

- Python 3.8+
- scikit-learn
- gensim
- nltk
- pandas, numpy
- tqdm
- matplotlib, seaborn (for figures)

---

## Citation

If you use the lexicon or methodology from this study, please cite:

> Pham, K. N. (2026). *Detection of exaggeration in English ESG disclosures: Evidence from Taiwanese listed firms via machine learning* [Master's thesis]. Tamkang University.

---

## Contact

For questions, data access requests, or collaboration inquiries, please contact:

**Pham Khac Nam**  
phamkhacnam2001@gmail.com
