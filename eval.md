# V3 — RAG Evaluation Baseline

## Overview

Version 3 is not a feature release.

V3 is the first evaluation-focused version of the RAG system. The purpose of this version was to move from:

> “The application sometimes gives good answers and sometimes gives bad answers.”

to:

> “I can repeatedly test the system and measure how often its generated answers are correct.”

The main goal was to build an evaluation loop around the RAG system without using LangChain, RAGAS, DeepEval, or another evaluation framework.

The evaluation was built manually in Python to understand the underlying mechanics.

---

## Evolution from V2

### V2 — Working RAG Application

V2 was an interactive Streamlit application.

The pipeline was:

```text
PDF
 ↓
Text extraction
 ↓
Semantic chunking
 ↓
Sentence Transformer embeddings
 ↓
ChromaDB
 ↓
User question
 ↓
Question embedding
 ↓
Top-k retrieval
 ↓
Gemini
 ↓
Generated answer
```

V2 successfully demonstrated that the complete RAG pipeline worked.

However, there was an important limitation.

When a question was answered incorrectly, it was not immediately clear **why**.

There were two possible explanations:

1. The correct information was not retrieved.
2. The correct information was retrieved, but the LLM generated a bad answer.

V2 did not measure these separately.

---

# The V3 Problem

The central question for V3 became:

> **How can I systematically evaluate whether my RAG system is producing correct answers?**

Instead of manually asking questions through the Streamlit UI, V3 uses a fixed evaluation dataset and repeatedly runs the same questions through the system.

The UI was therefore removed.

V3 is a static evaluation program rather than an interactive application.

---

# Test Document

The evaluation document used for V3 is:

**Afcons Infrastructure Limited — Q1 FY26 Earnings Conference Call**

The document contains the company's earnings-call presentation and analyst Q&A.

The PDF used for evaluation is approximately 19 pages long.

The Q&A section contains questions covering areas such as:

* Revenue and EBITDA
* Order book
* Order-flow guidance
* Project pipeline
* Capex
* Debt
* Interest costs
* Croatia projects
* Jal Jeevan Mission
* High-speed rail
* New business segments

---

# V3 Architecture

The Streamlit interface from V2 was removed.

The V3 evaluation pipeline is:

```text
Fixed PDF
   ↓
Text Extraction
   ↓
Semantic Chunking
   ↓
Sentence Transformer
   ↓
Document Embeddings
   ↓
ChromaDB
   ↓
Evaluation Questions
   ↓
Question Embedding
   ↓
ChromaDB Top-k Retrieval
   ↓
Retrieved Context
   ↓
LLM #1
   ↓
Generated Answer
   ↓
LLM #2
   ↓
Answer Correctness
   ↓
Aggregate Score
```

---

# Evaluation Dataset

The evaluation dataset is stored as a Python list of dictionaries.

Each evaluation question contains:

```python
{
    "id": "...",
    "question": "...",
    "expected_answer": "...",
    "ground_truth_evidence": "..."
}
```

### Meaning of the fields

### `id`

A unique identifier for the evaluation question.

For example:

```text
id1
id2
id3
```

These IDs identify **questions**, not document chunks.

### `question`

The question given to the RAG system.

### `expected_answer`

The reference answer used by the evaluator.

### `ground_truth_evidence`

The evidence from the source document that supports the expected answer.

This is useful because the evaluation should ultimately be grounded in the original document rather than in the model's assumptions.

---

# Example Evaluation Question

```python
{
    "id": "id1",
    "question": "What was Afcons' total income in Q1 FY26?",
    "expected_answer": "INR 3,419 crores.",
    "ground_truth_evidence":
        "Afcons reported a total income of INR3,419 crores in Q1 FY26."
}
```

The same structure is repeated for the evaluation questions.

The initial V3 run used five active questions to avoid unnecessarily consuming the available Gemini API quota while the evaluation pipeline was still being developed.

---

# Retrieval

For every evaluation question, the question is converted into an embedding using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The embedding is then passed to ChromaDB.

The current retrieval configuration uses:

```text
n_results = 3
```

Therefore ChromaDB returns the three closest document chunks according to vector similarity.

Conceptually:

```text
Question
   ↓
Question Embedding
   ↓
ChromaDB
   ↓
Top 3 Chunks
```

The retrieved chunk text is then combined and supplied to the first LLM.

---

# LLM #1 — Answer Generation

The first Gemini call is responsible only for answering the question.

It receives:

```text
Question
+
Retrieved Context
```

The model is instructed to answer using only the retrieved context.

The generated answer is stored in a variable:

```python
llm_op
```

The important separation here is:

> LLM #1 generates the answer. It does not evaluate itself.

---

# LLM #2 — Answer Evaluation

A second Gemini call is used as the evaluator.

This evaluator receives information including:

```text
Question
Expected Answer
LLM #1 Answer
Retrieved Context
Ground-truth Evidence
```

Its job is to determine whether the answer produced by LLM #1 is correct.

The evaluator is instructed to return only:

```text
1
```

for a correct answer, or:

```text
0
```

for an incorrect answer.

This deliberately keeps the output machine-readable so it can be aggregated directly in Python.

---

# Answer Correctness Metric

For each question:

```text
Correct → 1
Incorrect → 0
```

The scores are summed.

For `N` evaluated questions:

```text
Answer Correctness =
(sum of correctness scores / N) × 100
```

For example:

```text
Question 1 → 1
Question 2 → 1
Question 3 → 0
Question 4 → 0
Question 5 → 1
```

Total:

```text
3 / 5
```

Therefore:

```text
Answer Correctness = 60%
```

---

# First V3 Baseline

The first five-question evaluation produced:

```text
1
1
0
0
1
```

Therefore:

```text
Correct = 3
Incorrect = 2

Answer Correctness = 60%
```

This became the initial V3 baseline.

The purpose of this number is not to claim that the RAG system is “60% accurate” in a general sense.

It is a baseline for this specific evaluation set and configuration.

---

# Important Limitation of V3

The original motivation for V3 was to distinguish two possible RAG failure modes:

```text
Retrieval Failure
```

and:

```text
Generation Failure
```

The current V3 implementation successfully measures **answer correctness**, but retrieval hit-rate measurement is not yet fully integrated into the final score.

The next step is therefore to explicitly record:

```text
Ground-truth chunk IDs
```

and compare them against:

```text
Retrieved chunk IDs
```

For example:

```text
Ground Truth:
[id7]

Retrieved:
[id12, id7, id31]
```

This would produce:

```text
Retrieval = HIT
```

while:

```text
Ground Truth:
[id7]

Retrieved:
[id12, id31, id45]
```

would produce:

```text
Retrieval = MISS
```

This separation is important because:

```text
Retrieval HIT + Answer WRONG
```

suggests a generation problem.

Whereas:

```text
Retrieval MISS + Answer WRONG
```

suggests a retrieval problem.

---

# Why V3 Matters

V3 represents a change in how the system is developed.

Instead of changing the RAG pipeline based on anecdotal observations, the system now has a repeatable evaluation process.

The development loop becomes:

```text
Run evaluation
      ↓
Measure
      ↓
Identify failure
      ↓
Change system
      ↓
Run evaluation again
      ↓
Compare results
```

This turns RAG development into an experiment rather than trial-and-error debugging.

---

# What I Learned From V3

The main learning from V3 was that building a RAG system and evaluating a RAG system are two different engineering problems.

A working RAG pipeline answers questions.

An evaluation pipeline answers a different question:

> **How well is the RAG pipeline actually working, and where is it failing?**

V3 established the first measurement layer.

---

# V4 Direction

V4 will focus on **orchestration and diagnosis**.

The goal is to move beyond a single overall answer-correctness percentage and make the evaluation pipeline expose the intermediate stages of each question.

The intended direction is:

```text
Question
   ↓
Question Embedding
   ↓
Chroma Retrieval
   ↓
Retrieved Chunk IDs
   ↓
Retrieval Evaluation
   ↓
Retrieved Context
   ↓
LLM #1
   ↓
Generated Answer
   ↓
LLM #2
   ↓
Answer Evaluation
   ↓
Per-question Result
```

The final V4 result should make it possible to inspect where a question failed rather than only seeing an aggregate percentage.

---

# Version Philosophy

The versions are intentionally being developed incrementally:

```text
V1
Build the RAG plumbing.

V2
Turn the pipeline into a usable interactive application.

V3
Build the evaluation baseline.

V4
Orchestrate and diagnose the failure modes.

V5
Change the underlying RAG system based on measured failures
and demonstrate that the changes improve the benchmark.
```

The purpose of versioning is therefore not to continuously add features.

Each version represents a new engineering capability learned from the previous version.

---

# Current Status

**V3 Status: Evaluation Baseline Complete**

Implemented:

* Fixed evaluation document
* Manual evaluation questionnaire
* Semantic document chunking
* Sentence Transformer embeddings
* ChromaDB retrieval
* Fixed evaluation loop
* LLM #1 answer generation
* LLM #2 answer evaluation
* Binary answer correctness scoring
* Aggregate percentage calculation
* Initial five-question baseline

Current baseline:

```text
Answer Correctness: 60%
Evaluation Size: 5 questions
```

Next major capability:

```text
Retrieval Hit Rate
+
Failure Classification
+
Orchestration
```

That work forms the basis of V4.
