# 01 · Question ↔ Answer Retrieval with Multilingual Embeddings

**Task:** 1,000 random question–answer pairs are taken from [`ytu-ce-cosmos/gsm8k_tr`](https://huggingface.co/datasets/ytu-ce-cosmos/gsm8k_tr), a Turkish version of GSM8K with math word problems and their worked solutions. Each question has to be matched with its own answer among all 1,000, and the other way round. Candidates are ranked by **angular distance** between embeddings.

## Results

| Model | Top-1 Q→A | Top-5 Q→A | Top-1 A→Q | Top-5 A→Q | Embedding time |
|---|---|---|---|---|---|
| **BAAI/bge-m3** | **94.7%** | **98.1%** | **97.1%** | **98.5%** | 1,166 s |
| intfloat/multilingual-e5-base | 94.3% | 96.9% | 96.1% | 98.4% | 356 s |
| sentence-transformers/static-similarity-mrl-multilingual-v1 | 93.3% | 97.2% | 88.1% | 95.1% | **0.2 s** |
| sentence-transformers/LaBSE | 91.3% | 95.0% | 95.0% | 97.2% | 317 s |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 89.0% | 95.9% | 88.3% | 95.0% | 118 s |
| ytu-ce-cosmos/turkish-colbert | 79.2% | 91.2% | 82.6% | 93.0% | 276 s |
| HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1 | 77.2% | 79.6% | 67.3% | 70.7% | 3,125 s |

*Embedding time for 2,000 texts (1,000 questions + 1,000 answers).*

## Takeaways

- **bge-m3 and multilingual-e5-base are the most accurate**, and e5-base is about 3× faster.
- **The static embedding model is a strong speed/quality trade-off.** It reaches 93% Top-1 in 0.2 seconds, which is hundreds of times faster than the transformer models.
- The task is fairly easy because a question and its solution share many surface tokens such as numbers and names. That explains why even small models score above 75%.

## Notebook

[`embedding_benchmark.ipynb`](embedding_benchmark.ipynb): embedding extraction with caching, Top-k evaluation in both directions, bar charts and t-SNE plots.
