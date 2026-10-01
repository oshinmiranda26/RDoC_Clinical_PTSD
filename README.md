# RDoC-PTSD-NLP: Extracting Research Domain Criteria from Clinical Notes

An NLP pipeline that identifies **Research Domain Criteria (RDoC)** signals in clinical notes of patients with post-traumatic stress disorder (PTSD), combining keyword filtering with transformer-based semantic similarity. This repository contains a cleaned, runnable version of the research code behind the study cited below, demonstrated on **fully synthetic notes**.

## Why it matters

RDoC is a framework from the National Institute of Mental Health that describes mental health along functional domains (for example, negative valence, arousal and regulatory systems, and sensorimotor systems) rather than diagnostic categories alone. Much of this information appears only in free-text clinical notes. Extracting it at scale makes it possible to study how these domains differ across patient populations and change over the course of illness and treatment.

## How the pipeline works

```
Clinical notes (ENC1.csv, ENC2.csv)
        |
Group notes by patient; split each note into sentences (NLTK)
        |
Stage 1 - Keyword filter: keep sentences containing an RDoC dictionary keyword
        |
Stage 2 - Semantic match: embed each candidate sentence and its dictionary
          example sentence with all-mpnet-base-v2 (sentence-transformers)
        |
Keep pairs with cosine similarity >= 0.3
        |
Output: patient, note, date, keyword, RDoC domain, similarity score
```

The two-stage design keeps computation manageable: the inexpensive keyword filter narrows the candidates, and the transformer model is applied only to those sentences to confirm that their meaning matches the dictionary example.

## Published study

In the original study, dictionaries were built from 100,000 clinical notes, and the pipeline was applied to 5.67 million clinical notes from 38,807 patients with PTSD at the University of Pittsburgh Medical Center. A cosine similarity threshold of 0.3 achieved an F1 score of at least 0.80 across all six RDoC domains.

## Data in this repository

- `ENC1.csv`, `ENC2.csv`: **fully synthetic** clinical notes, generated from template sentences to mirror the structure of the study's encounter notes. They contain no patient data. See [DATA_NOTICE.md](DATA_NOTICE.md).
- `RDoC_Dictionary_subset.csv`: a subset of the RDoC sentence dictionary developed for the study (keyword, RDoC domain, example sentence).
- `RDoC_keyword_dictionary.csv`: the RDoC keyword dictionary subset.

On the synthetic notes, the pipeline produces 228 matches spanning all six RDoC domains (see `results.csv`). These results demonstrate the pipeline only; they are not research findings.

## How to run

```bash
pip install -r requirements.txt
python run.py
```

The first run downloads the all-mpnet-base-v2 model (about 420 MB) and NLTK's sentence tokenizer data.

**Outputs**
- `results.csv`: one row per match, with columns `STUDY_ID`, `STUDY_NOTE_ID`, `CONTACT_DATE`, `KEYWORD`, `RDOC_DOMAIN`, `SIMILARITY`
- `result.pkl`: the same matches as a Python pickle (original format)

## Changes from the original research code

- Restored the sentence-transformers model definition (all-mpnet-base-v2) that was missing from the earlier public version
- Replaced the deprecated `DataFrame.append` with `pd.concat` for pandas 2.x compatibility
- Made keyword matching case-insensitive, and fixed an off-by-one check that skipped keywords at the start of a note or sentence
- Fixed silent truncation of note text: notes were read with a pandas display function that cut each note to 50 characters; full note text is now analyzed
- Processed all patients instead of a single batch from the original parallel run
- Replaced hard-coded file paths with paths relative to the repository
- Added a human-readable CSV output
- Replaced patient notes with fully synthetic notes

## Limitations

- Keyword filtering means sentences that express an RDoC concept without any dictionary keyword are not detected
- A single similarity threshold is applied across all domains
- The dictionary here is a subset; coverage is narrower than in the full study

## Citation

If you use this code, please cite:

Miranda O, et al. Enhancing post-traumatic stress disorder patient assessment: Leveraging Natural Language Processing for Research of Domain Criteria Identification using electronic medical records. *Research Square* (preprint), 2024. doi:10.21203/rs.3.rs-3973337/v1

## Author

**Oshin Miranda, PhD** | [LinkedIn](https://www.linkedin.com/in/oshin-miranda-ph-d-9551781b5/) | [Google Scholar](https://scholar.google.com/citations?hl=en&user=fkvhYbgAAAAJ&view_op=list_works&sortby=pubdate)
