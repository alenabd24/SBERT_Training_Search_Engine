# MS MARCO Passage Ranking with SBERT

This repository implements a training and evaluation pipeline for a re-ranking model using the MS MARCO Passage Ranking dataset. The goal is to fine-tune a Sentence-BERT (SBERT) model to score candidate passages in response to a query, distinguishing relevant from non-relevant results based on learned semantic representations.

The MS MARCO dataset is widely used for benchmarking passage retrieval and re-ranking models in information retrieval (IR) research.

---

## Task Overview

Given a user query, the task is to re-rank a list of candidate passages so that relevant passages appear at the top. This is a supervised learning problem where relevance labels are provided for training, and model performance is evaluated on a separate development set using standard IR metrics.

The model is trained using **pointwise learning**: the SBERT model takes a (query, passage) pair and learns to classify it as relevant or not based on binary labels from the training set.

---

## Dataset Structure

The dataset is composed of several key files, each with a specific role in the pipeline:

- **`collection.tsv`**  
  Contains all passages in the corpus. Each line is formatted as:  
  `pid<TAB>passage_text`  
  This allows passage text to be retrieved using the passage ID.

- **`queries.train.tsv` / `queries.dev.tsv`**  
  These files contain training and development (validation) queries, formatted as:  
  `qid<TAB>query_text`

- **`qrels.train.tsv`**  
  Contains the relevance labels for training, in the TREC format:  
  `qid 0 pid relevance_label`  
  These labels are used to form (query, passage, label) triples for training the model.

- **`qrels.dev.tsv`**  
  Contains the ground-truth relevance labels for development queries. This is used to evaluate the model’s ability to generalize to unseen queries.

---

## Pipeline Summary

The notebook implements the following steps:

1. **Preprocessing**  
   - Load passages into a dictionary: `pid → passage_text`  
   - Load queries into a dictionary: `qid → query_text`  
   - Parse `qrels.train` to generate positive training pairs

2. **Candidate Generation**  
   - Candidate (qid, pid) pairs may be pre-generated or sampled during training
   - Unlabeled pairs are treated as negatives (label = 0)

3. **Training**  
   - Fine-tune a pretrained SBERT model using labeled (query, passage) pairs  
   - Optimized using binary cross-entropy or contrastive loss

4. **Evaluation**  
   - Use `qrels.dev` and `queries.dev` to compute ranking metrics such as:
     - **MRR (Mean Reciprocal Rank)**
     - **nDCG@k (Normalized Discounted Cumulative Gain)**  
   - Evaluate the model's ability to rank truly relevant passages higher

---

## Objective

By fine-tuning SBERT on the MS MARCO Passage Ranking dataset, this project aims to:

- Improve the ranking of relevant passages over traditional lexical methods
- Leverage semantic embeddings for more meaningful query-passage matching
- Provide a robust and reproducible pipeline for IR model training and evaluation

---

## Prerequisites

- Python 3.7+
- HuggingFace Transformers
- SentenceTransformers
- TQDM, NumPy, Pandas, Scikit-learn

---

## Future Extensions

- Incorporate **hard negatives** from BM25 or other retrievers
- Switch to **pairwise or listwise loss functions**
- Add support for evaluation on **TREC DL** or **BEIR** benchmarks
- Use **ColBERT-style** late interaction architectures for improved performance

---

## References

- [MS MARCO Dataset](https://microsoft.github.io/msmarco/)
- [SBERT: Sentence-BERT](https://www.sbert.net/)
- [TREC Ranking Guidelines](https://trec.nist.gov/)
