# NLP & ML Experiments

A collection of smaller experiments from my graduate coursework: embeddings, LLM prompting, text classification and optimization. Each folder is self-contained and has its own notebook and README.

| # | Experiment | Highlights | Key result |
|---|---|---|---|
| 01 | [Embedding benchmark on Turkish GSM8K](01-embedding-benchmark-gsm8k-tr/) | 7 multilingual embedding models, bidirectional question ↔ answer retrieval | **bge-m3: 94.7% Top-1** (Q→A) |
| 02 | [Few-shot prompting of Turkish LLMs](02-llm-few-shot-trglue/) | 4 open 7–9B LLMs, 0/3/5-shot on TrGLUE SST-2 and CoLA, prompt and parser iteration | Improved prompt: **SST-2 up to 80.5%**, **CoLA up to 85.7%** |
| 03 | [News & tweet classification with embeddings](03-text-classification-e5-jina/) | E5 vs. Jina embeddings, concatenation, 4 classifiers, feature-space expansion | **Jina + LogReg: 80.8% (news, 7 classes) / 84.3% (tweets)** |
| 04 | [Optimizer comparison on IMDB](04-optimizer-comparison-imdb/) | SGD vs. momentum vs. Adam, 5 random initializations each, grid search, t-SNE of weight trajectories | **Adam 85.1%** vs. SGD 62.3% |

The flagship projects are in separate repositories:

- [llm-attention-sentiment-analysis](https://github.com/ezgiieyice/llm-attention-sentiment-analysis): attention matrices of BERT and Gemma-9B used as features for sentiment analysis
- [turkish-gpt2-lora-finetuning](https://github.com/ezgiieyice/turkish-gpt2-lora-finetuning): LoRA instruction tuning with GPT-4o vs. DeepSeek answers as training data
- [ensemble-embeddings-semantic-search](https://github.com/ezgiieyice/ensemble-embeddings-semantic-search): ensembles of embedding models for retrieval

## Setup

```bash
git clone https://github.com/ezgiieyice/nlp-experiments.git
cd nlp-experiments
pip install -r requirements.txt
```

Experiment 02 needs a GPU (7–9B models) and a Hugging Face account for the gated models. Experiment 03 calls the Jina embeddings API, so set `JINA_API_KEY` in your environment first.

---

*Coursework for the graduate courses **Computational Semantics** (01, 02) and **Ensemble Learning** (03, 04) at Yıldız Technical University.*
