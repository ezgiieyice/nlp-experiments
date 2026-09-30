# 04 · Optimizer and Initialization Comparison on IMDB

A feed-forward network (TensorFlow/Keras) is trained on IMDB movie-review sentiment with three optimizers. Each optimizer is started from **5 different random initializations** to see how much the starting point matters. A grid search then tunes each optimizer's hyperparameters.

## Best configuration per optimizer

| Optimizer | Best hyperparameters | Accuracy |
|---|---|---|
| SGD | lr = 0.1 | 62.3% |
| SGD + momentum | lr = 0.01, momentum = 0.9 | 83.2% |
| **Adam** | lr = 0.001, β₁ = 0.9, β₂ = 0.9999 | **85.1%** |

## Takeaways

- **Plain SGD struggles on this problem.** It needs a large learning rate and still trails by more than 20 points. Adding momentum closes most of the gap.
- **Adam works best with a small learning rate** (0.001) and high β values.
- **The random initialization affects training speed as well as accuracy.** For SGD, the average epoch time ranged from 2.67 s to 3.44 s depending on the seed.
- The notebook also visualizes the **weight trajectories** of all 15 runs with t-SNE, which shows how differently the optimizers move through the weight space.

## Notebook

[`optimizer_comparison.ipynb`](optimizer_comparison.ipynb): runs on a CPU; IMDB is loaded via `tensorflow-datasets`.
