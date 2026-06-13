# Experiment 1 — Iris Setosa vs Non-Setosa

## 1. Objective

The objective of this experiment was to verify the correctness of a manually implemented binary logistic regression pipeline.

Rather than using a ready-made logistic regression model, the core learning algorithm was implemented from scratch using NumPy.

This experiment tested the complete workflow:

* loading and preparing the dataset
* converting a multiclass target into binary labels
* creating training, cross-validation, and test splits
* preventing data leakage during feature scaling
* manually implementing the sigmoid function
* manually calculating binary cross-entropy cost
* manually calculating gradients
* training using gradient descent
* generating binary predictions
* manually calculating evaluation metrics

The primary purpose of this experiment was implementation verification and mathematical understanding, not solving a difficult classification problem.

---

## 2. Problem Definition

The original Iris dataset contains three flower species:

* `0` — Iris Setosa
* `1` — Iris Versicolor
* `2` — Iris Virginica

For this experiment, the target was converted into a binary classification problem:

* `1` — Setosa
* `0` — Non-Setosa

The model therefore predicts whether a given flower belongs to the Setosa species.

---

## 3. Input Features

Each flower is represented using four numerical features:

1. Sepal length
2. Sepal width
3. Petal length
4. Petal width

The input matrix has the form:

```text
X.shape = (150, 4)
```

This represents:

* 150 flower samples
* 4 features per sample

---

## 4. Tools Used

* Python
* NumPy
* scikit-learn dataset loader
* scikit-learn train-test splitting utility

Scikit-learn was used only for:

* loading the Iris dataset
* randomly splitting the dataset

The logistic regression model itself was not trained using scikit-learn.

---

## 5. Data Splitting

The dataset was divided into three parts:

| Split                | Percentage | Purpose                                                  |
| :------------------- | ---------: | :------------------------------------------------------- |
| Training set         |        70% | Used to calculate gradients and learn model parameters   |
| Cross-validation set |        15% | Used to observe model performance and training behaviour |
| Test set             |        15% | Used only for final model evaluation                     |

The approximate split sizes were:

```text
Training examples: 105
Cross-validation examples: 22
Test examples: 23
```

Stratified splitting was used so that each split retained a similar proportion of Setosa and Non-Setosa examples.

A fixed random state was used to make the experiment reproducible.

---

## 6. Feature Scaling

Z-score normalization was applied to all four input features.

The normalization formula was:

```text
X_scaled = (X - mean) / standard_deviation
```

To prevent data leakage:

* the mean was calculated using only the training set
* the standard deviation was calculated using only the training set
* the same training statistics were used to transform the training, validation, and test sets

Separate means and standard deviations were not calculated for the validation or test sets.

---

## 7. Logistic Regression Model

The linear component of the model is:

```text
z = w · x + b
```

The sigmoid function converts the linear score into a probability:

```text
prediction_probability = 1 / (1 + e^(-z))
```

The final binary prediction is determined using a threshold of `0.5`:

```text
probability >= 0.5 → Setosa
probability < 0.5  → Non-Setosa
```

---

## 8. Parameter Initialization

The model was initialized using:

```text
Initial weights: [0, 0, 0, 0]
Initial bias: 0
```

With zero weights and zero bias, every initial prediction is:

```text
sigmoid(0) = 0.5
```

Therefore, the initial binary cross-entropy cost was approximately:

```text
0.6931
```

---

## 9. Cost Function

Binary cross-entropy was implemented manually.

For one training example:

```text
loss = -y log(y_hat) - (1-y) log(1-y_hat)
```

The total cost was calculated by averaging the loss across all training examples.

To avoid numerical errors from evaluating `log(0)`, predicted probabilities were clipped to a small interval near zero and one before applying the logarithm.

---

## 10. Gradient Calculation

The prediction error for each example was calculated as:

```text
error = predicted_probability - actual_label
```

The gradient for each feature weight was calculated using:

```text
dj_dw[j] = average(error × X[j])
```

The bias gradient was calculated using:

```text
dj_db = average(error)
```

The parameters were then updated using gradient descent:

```text
w = w - alpha × dj_dw
b = b - alpha × dj_db
```

---

## 11. Learning-Rate Experiment

Several learning rates were tested using 1,000 gradient-descent updates.

| Alpha | Initial Cost | Final Cost | CV Accuracy | Observation                            |
| ----: | -----------: | ---------: | ----------: | :------------------------------------- |
| 0.001 |       0.6931 |     0.3847 |        100% | Stable, very slow                      |
|  0.01 |       0.6931 |     0.0837 |        100% | Stable, faster                         |
|   0.1 |       0.6931 |     0.0109 |        100% | Fast and stable                        |
|   1.0 |       0.6931 |     0.0011 |        100% | Very fast and stable                   |
|   3.0 |       0.6931 |     0.0004 |        100% | Extremely fast                         |
|  10.0 |       0.6931 |     Near 0 |        100% | Near-saturated probabilities           |
|  100+ |       0.6931 |  Approx. 0 |        100% | Numerical saturation; no accuracy gain |

---

## 12. Interpretation of the Learning-Rate Results

All tested learning rates achieved 100% validation accuracy.

This occurred because Setosa is strongly linearly separable from the other two Iris species, particularly through petal length and petal width.

Once every sample was placed on the correct side of the decision boundary, increasing the learning rate mainly caused the model to become more confident.

For example:

```text
Correct probability: 0.90
Correct probability: 0.99
Correct probability: 0.999999
```

All three probabilities produce the same final class after applying the `0.5` threshold.

Therefore, a smaller training cost did not necessarily represent better classification performance.

Extremely large learning rates produced large parameter values and sigmoid outputs extremely close to exact zero or one. This reduced the numerical loss but did not improve accuracy or generalization.

---

## 13. Selected Baseline Configuration

The following configuration was retained for the final baseline experiment:

```text
Learning rate: 1.0
Gradient-descent iterations: 1000
Classification threshold: 0.5
```

This value was selected because it produced fast and stable convergence without relying on unnecessarily extreme parameter updates.

The goal was not to achieve the smallest representable training cost. The goal was to obtain a correct, stable, and understandable baseline implementation.

---

## 14. Final Evaluation Results

The manually implemented model achieved perfect classification on the selected splits.

| Dataset Split    | Accuracy |
| :--------------- | -------: |
| Training         |     100% |
| Cross-validation |     100% |
| Test             |     100% |

The final test-set classification metrics were:

| Metric    | Result |
| :-------- | -----: |
| Accuracy  |   1.00 |
| Precision |   1.00 |
| Recall    |   1.00 |
| F1 score  |   1.00 |

---

## 15. Test-Set Confusion Matrix

The test set contained 23 examples.

The manually calculated confusion-matrix values were:

| Value           | Count |
| :-------------- | ----: |
| True Negatives  |    15 |
| False Positives |     0 |
| False Negatives |     0 |
| True Positives  |     8 |

The confusion matrix was therefore:

|                   | Predicted Non-Setosa | Predicted Setosa |
| :---------------- | -------------------: | ---------------: |
| Actual Non-Setosa |                   15 |                0 |
| Actual Setosa     |                    0 |                8 |

The model made no incorrect predictions on the test split.

---

## 16. Interpretation of the Result

The 100% test accuracy is plausible because Setosa is easily separated from Versicolor and Virginica using the available numerical features.

Setosa flowers generally have much smaller petal lengths and petal widths.

After Z-score normalization:

* Setosa examples usually had strongly negative petal measurements
* Non-Setosa examples usually had larger standardized petal measurements

The model learned negative weights for important petal-related features.

Because Setosa samples had negative standardized petal measurements, multiplying those measurements by negative weights increased the linear score and pushed the predicted probability toward the Setosa class.

---

## 17. Why Accuracy Alone Was Not Informative

Accuracy reached 100% for almost every tested learning rate.

Because of this, accuracy could not meaningfully distinguish between the different training configurations.

The experiment instead revealed that:

* training cost measures confidence as well as correctness
* lower training cost does not always mean better classification
* extremely large learning rates can create saturated probabilities
* numerical stability must be considered when implementing sigmoid and logarithmic loss manually
* hyperparameter selection is difficult on an overly simple and strongly separable task

---

## 18. Limitations

This experiment has several limitations:

1. The classification problem is very easy.
2. Setosa is strongly linearly separable from the other two species.
3. The dataset contains only 150 total examples.
4. The validation and test sets are small.
5. The model uses no regularization.
6. Perfect accuracy prevents meaningful error analysis.
7. The learning-rate comparison mostly measured convergence speed and confidence rather than differences in classification quality.

The result should therefore not be interpreted as proof that the model will achieve perfect accuracy on all unseen real-world flower data.

---

## 19. Main Learnings

This experiment provided practical experience with:

* NumPy array shapes
* matrix and vector multiplication
* binary target conversion
* stratified dataset splitting
* avoiding data leakage
* Z-score normalization
* sigmoid activation
* binary cross-entropy
* manual gradient calculation
* gradient descent
* learning-rate behaviour
* sigmoid saturation
* floating-point precision
* probability clipping
* manual confusion-matrix calculation
* manual accuracy, precision, recall, and F1 calculation
* organizing an ML project across multiple Python files
* documenting experimental decisions

---

## 20. Conclusion

Experiment 1 successfully verified that the manually implemented logistic regression pipeline works correctly.

The model:

* processed the dataset correctly
* learned its parameters through gradient descent
* reduced binary cross-entropy cost
* produced correct binary predictions
* achieved perfect performance on the training, validation, and test splits

However, Setosa vs Non-Setosa was too easy for meaningful model comparison, hyperparameter tuning, or error analysis.

For this reason, the same manually implemented logistic regression pipeline will next be evaluated on a more difficult task:

```text
Iris Versicolor vs Iris Virginica
```

These two species overlap more significantly, making Experiment 2 more suitable for:

* learning-rate selection
* validation-based tuning
* confusion-matrix analysis
* precision and recall interpretation
* F1-score evaluation
* misclassification analysis
* honest model limitations