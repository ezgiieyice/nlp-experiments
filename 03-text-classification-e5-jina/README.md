# 03 · Turkish News & Tweet Classification with E5 and Jina Embeddings

Embeddings from two models are used as features for classic classifiers. The two datasets are very different kinds of Turkish text:

| Dataset | Task | Samples |
|---|---|---|
| [`yankihue/turkish-news-categories`](https://huggingface.co/datasets/yankihue/turkish-news-categories) | News headline topic (7 classes, formal and clean) | 5,397 (balanced, 771 per class) |
| [`yankihue/tweets-turkish`](https://huggingface.co/datasets/yankihue/tweets-turkish) | Tweet sentiment (binary, informal and noisy) | 6,000 (balanced) |

The representations are `intfloat/e5-small` (384-d), Jina embeddings via API (1,024-d), and the **concatenation** of both (1,408-d).

## Results (test accuracy)

| Classifier | News: E5 | News: Jina | News: concat | Tweets: E5 | Tweets: Jina | Tweets: concat |
|---|---|---|---|---|---|---|
| Logistic Regression | 51.9% | **80.8%** | 78.7% | 74.0% | 84.3% | **84.8%** |
| SVM | 51.7% | 79.5% | 76.2% | 72.3% | 84.3% | 83.3% |
| Random Forest | 46.3% | 76.9% | 79.1% | 69.4% | 82.3% | 82.0% |
| KNN | 53.1% | 76.6% | 75.6% | 69.1% | 75.8% | 75.4% |

An additional **"improved space"** experiment expands the concatenated embeddings 1.5× to 5× with new features built from random feature subsets. This gave news accuracies of about 74–78% and tweet accuracies up to 85.2% (Logistic Regression at 1.5×), so there was no clear gain over the plain Jina embeddings.

## Takeaways

- **The choice of embedding model matters far more than the classifier.** Jina is 25–30 points better than E5-small on the 7-class news task.
- **Concatenation only helps when both parts are informative.** On news, the weak E5 features slightly *hurt* the linear models. On tweets, where E5 is reasonable, concatenation gives the best overall result.
- Linear models (Logistic Regression, SVM) are competitive with or better than Random Forest and KNN on dense sentence embeddings.

## Notebook

[`text_classification_embeddings.ipynb`](text_classification_embeddings.ipynb) requires the environment variable `JINA_API_KEY` for the Jina embedding calls.
