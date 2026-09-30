# 02 · Zero- and Few-Shot Prompting of Turkish LLMs on TrGLUE

Four open 7–9B instruction models are evaluated on two tasks from [TrGLUE](https://huggingface.co/datasets/turkish-nlp-suite/TrGLUE) with **0, 3 and 5 in-context examples**. Each run uses 200 test sentences.

- **SST-2:** sentiment (positive / negative)
- **CoLA:** grammatical acceptability (acceptable / not acceptable)

| Short name | Model |
|---|---|
| turkcell | [`TURKCELL/Turkcell-LLM-7b-v1`](https://huggingface.co/TURKCELL/Turkcell-LLM-7b-v1) |
| cosmos | [`ytu-ce-cosmos/Turkish-Llama-8b-DPO-v0.1`](https://huggingface.co/ytu-ce-cosmos/Turkish-Llama-8b-DPO-v0.1) |
| koc | [`KOCDIGITAL/Kocdigital-LLM-8b-v0.1`](https://huggingface.co/KOCDIGITAL/Kocdigital-LLM-8b-v0.1) |
| gemma | [`google/gemma-2-9b-it`](https://huggingface.co/google/gemma-2-9b-it) |

## Round 1 · Baseline prompt

| Model | SST-2 0-shot | 3-shot | 5-shot | CoLA 0-shot | 3-shot | 5-shot |
|---|---|---|---|---|---|---|
| turkcell | 74.1% | **75.8%** | 72.7% | 39.0% | 41.0% | 41.5% |
| cosmos | 74.5% | 73.3% | 73.4% | 39.0% | 38.5% | 39.0% |
| koc | 74.5% | 71.9% | 75.0% | 39.0% | 38.5% | **42.0%** |
| gemma | 74.5% | 72.4% | 74.4% | 39.0% | 41.0% | 40.5% |

All four models land on almost identical numbers, and CoLA accuracy sits around 39%. The label distribution explains this: the test data is **imbalanced**, and the models mostly predict a single class. Accuracy then mirrors the class ratio rather than the models' actual ability.

## Round 2 · Improved prompt, stricter output parsing, balanced data

| Model | SST-2 0-shot | 3-shot | 5-shot | CoLA 0-shot | 3-shot | 5-shot |
|---|---|---|---|---|---|---|
| gemma | 79.3% | 79.9% | 65.1% | **85.7%** | 63.0% | 65.0% |
| cosmos | **80.5%** | 76.1% | 67.0% | 0.0%* | 56.0% | 57.5% |
| turkcell | 0.0%* | 69.8% | 55.4% | 52.3% | 53.0% | 56.3% |
| koc | 0.0%* | 69.5% | 64.3% | 0.0%* | 0.0%* | – |

\* 0% means the model's answers could not be parsed into a label. The model ignored the requested answer format. It does not mean every prediction was wrong.

## Takeaways

- **Class balance matters more than the prompt.** On imbalanced data, "always answer the majority class" looks like a decent model: accuracy is high but F1 is low.
- **Gemma-2-9B follows instructions most reliably.** It is the only model that produced parseable answers in every setting, and it has the best CoLA score (85.7% zero-shot).
- **More shots are not always better.** With 5 examples, accuracy often dropped, probably because longer prompts pull the models away from the short answer format.
- **Output parsing is part of the evaluation.** With the Turkish-tuned models, the gap between "correct" and "unparseable" was often larger than the gap between the models themselves. Constrained decoding or log-likelihood scoring of the label words would be more robust.

## Files

- [`00_data_sampling.ipynb`](00_data_sampling.ipynb): samples 5K train / 1K test sentences from TrGLUE (saved in [`data/`](data/))
- [`01_few_shot_prompting.ipynb`](01_few_shot_prompting.ipynb): prompts, output parsing, and both evaluation rounds (GPU required)
