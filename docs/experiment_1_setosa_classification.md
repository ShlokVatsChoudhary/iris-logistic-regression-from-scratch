# Experiment 1 — Setosa vs Non-Setosa Baseline

## Objective

The objective of this experiment was to verify that the manually implemented logistic regression pipeline works correctly.

This experiment was used to test:

* binary label conversion
* train/validation/test splitting
* Z-score normalization
* sigmoid prediction
* binary cross-entropy cost
* gradient calculation
* gradient descent
* final classification

The goal was not to find a globally optimal learning rate, because Setosa is strongly separable from the other Iris species.

## Problem Definition

The original Iris dataset contains three classes:

* `0` — Setosa
* `1` — Versicolor
* `2` — Virginica

For this experiment, the target was converted into:

* `1` — Setosa
* `0` — Non-Setosa

The model uses four numerical features:

1. Sepal length
2. Sepal width
3. Petal length
4. Petal width

## Fixed Setup

* Dataset: Iris
* Task: Setosa vs Non-Setosa
* Training split: 70%
* Validation split: 15%
* Test split: 15%
* Split method: Stratified random splitting
* Feature scaling: Z-score normalization
* Scaling statistics: Calculated from training data only
* Initial weights: `[0, 0, 0, 0]`
* Initial bias: `0`
* Classification threshold: `0.5`
* Gradient-descent iterations used for comparison: `1000`

## Learning-Rate Behaviour
| Alpha | Initial Cost | Final Training Cost | Validation Accuracy | Observation                                                              |
| ----: | -----------: | ------------------: | ------------------: | :----------------------------------------------------------------------- |
| 0.001 |       0.6931 |              0.3847 |                100% | Stable but very slow cost reduction                                      |
|  0.01 |       0.6931 |              0.0837 |                100% | Stable and faster than `0.001`                                           |
|   0.1 |       0.6931 |              0.0109 |                100% | Fast and stable convergence                                              |
|   1.0 |       0.6931 |              0.0011 |                100% | Very fast and stable convergence                                         |
|   3.0 |       0.6931 |              0.0004 |                100% | Extremely fast convergence                                               |
|  10.0 |       0.6931 |     Approximately 0 |                100% | Produced highly confident predictions and near-saturated sigmoid outputs |
|  100+ |       0.6931 |     Approximately 0 |                100% | Caused numerical saturation without improving classification accuracy    |

## Main Finding

Every tested learning rate produced 100% validation accuracy because Setosa is strongly linearly separable from Versicolor and Virginica, especially through petal length and petal width.

Increasing the learning rate continued to reduce binary cross-entropy loss, but this did not improve classification performance after accuracy had already reached 100%.

The lower loss mainly represented increasingly confident predictions:

* correct probability near `0.90`
* correct probability near `0.99`
* correct probability near `0.999999`

All of these probabilities produce the same final class after applying the `0.5` threshold.

Therefore, the smallest training cost was not treated as evidence of a better classifier.

## Numerical Behaviour

At extremely large learning rates, the weights became large enough for the sigmoid output to approach exact `0` or `1`.

This caused the loss to approach floating-point zero. In some earlier runs, the calculated loss became slightly negative due to numerical precision and the previous epsilon implementation.

The cost implementation was corrected by clipping probabilities before applying the logarithm.

## Selected Baseline Configuration

A moderate configuration was retained for the final baseline:

* Learning rate: `1.0`
* Iterations: To be selected using the earliest point at which validation performance stabilizes
* Classification threshold: `0.5`

The purpose of this selection is not to achieve the absolute smallest possible training loss. It is to obtain fast, stable training without unnecessarily extreme parameter values.

## Final Performance

| Dataset Split | Accuracy |
| ------------- | -------: |
| Training      |     100% |
| Validation    |     100% |
| Test          |     100% |

These results are plausible because Setosa is easily separated from the other two Iris species.

The validation and test sets are also small, so this result should not be interpreted as proof that the model will always achieve perfect accuracy on unseen real-world data.

## Conclusion

Experiment 1 successfully verified that the manually implemented logistic regression pipeline works correctly.

However, the task was too easy for meaningful hyperparameter comparison or error analysis. The model reached perfect classification across training, validation, and test data for a wide range of learning rates.

For this reason, a second and more difficult experiment will classify:

* Versicolor
* Virginica

These classes overlap more significantly and should provide a better test of model evaluation, hyperparameter selection, and error analysis.
