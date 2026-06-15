# Logistic Regression From Scratch on the Iris Dataset

## Project Summary

## 1. Project Overview

In this project, I implemented binary logistic regression from scratch using Python and NumPy.

My primary objective was not to maximize accuracy using a ready-made machine-learning framework. My objective was to understand and implement the complete flow of a binary classification project, including:

* dataset preparation
* feature normalization
* model initialization
* sigmoid activation
* binary cross-entropy cost
* gradient calculation
* gradient descent
* hyperparameter experimentation
* probability prediction
* classification
* evaluation metrics
* visualization
* debugging
* error analysis
* technical documentation

I used scikit-learn only for loading the Iris dataset and creating reproducible dataset splits. I implemented the logistic regression model manually instead of using sklearn's `LogisticRegression` class.

The project contains two experiments of different difficulty:

1. **Setosa vs Non-Setosa**
2. **Versicolor vs Virginica**

I reused the same core model, preprocessing functions, metric functions, and visualization utilities across both experiments.

Detailed reports are available separately:

* [Experiment 1: Setosa vs Non-Setosa](experiment_1_setosa_classification.md)
* [Experiment 2: Versicolor vs Virginica](experiment_2_versicolor_vs_virginica.md)

---

## 2. Project Objectives

I created this project to achieve the following goals:

* convert logistic regression theory into working code
* understand what happens internally during `model.fit()`
* build a complete train-validation-test workflow
* prevent data leakage during normalization
* understand the effects of learning rate and iteration count
* distinguish between probability loss and classification metrics
* implement evaluation metrics manually
* organize an ML project into reusable Python modules
* perform real debugging and error analysis
* document the entire development process professionally

I consider this a foundational machine-learning project. Its strength lies in implementation depth, experimentation, understanding, and documentation rather than dataset complexity.

---

## 3. Technology Used

I used:

* Python
* NumPy
* Matplotlib
* scikit-learn
* Git
* GitHub
* Markdown

I limited the use of scikit-learn to:

* loading the Iris dataset
* splitting the data using `train_test_split`

I manually implemented:

* Z-score normalization
* sigmoid function
* binary cross-entropy
* logistic regression gradients
* gradient descent
* predictions
* confusion matrix
* accuracy
* precision
* recall
* F1 score

---

## 4. Repository Structure

```text
iris-logistic-regression-from-scratch/
│
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── metrics.py
│   ├── visualization.py
│   ├── experiment_1_data.py
│   ├── experiment_2_data.py
│   ├── train_experiment_1.py
│   └── train_experiment_2.py
│
├── docs/
│   ├── experiment_1_setosa_classification.md
│   ├── experiment_2_versicolor_vs_virginica.md
│   └── project_summary.md
│
├── plots/
│   ├── experiment_1/
│   └── experiment_2/
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE

---

## 5. Source-Code Organization

### `preprocessing.py`

This file contains my shared feature-scaling functions:

* `z_scale_mean`
* `z_scale_std`
* `z_scale_normalization`

I calculate the feature mean and standard deviation using only the training set.

I then apply the training-derived statistics to the training, cross-validation, and test sets.

This prevents information from the validation or test data from leaking into model training.

---

### `model.py`

This file contains the core logistic regression functions that I implemented manually:

* `sigmoid`
* `compute_cost`
* `compute_gradient_logistic`
* `gradient_descent`
* `predict_logistic`
* `iterations_to__target`

I use these functions for:

* calculating predicted probabilities
* calculating binary cross-entropy
* calculating gradients for the weights and bias
* updating model parameters
* storing cost history
* producing final binary predictions
* checking the number of iterations required to reach a target cost

---

### `metrics.py`

This file contains the classification metrics that I implemented manually:

* `confusion_matrix`
* `precision_score`
* `recall_score`
* `manual_accuracy`
* `f1_score`

My confusion-matrix function returns:

* true negatives
* false positives
* false negatives
* true positives

I then use these values to calculate the remaining evaluation metrics.

---

### `visualization.py`

This file handles the project's visual diagnostics using custom Matplotlib implementations. Instead of manually writing plotting scripts inside the training loops, this module exposes reusable functions to consistently evaluate model performance:

* `plot_binary_cross_entropy_cost`
* `plot_confusion_matrix`

I use these functions to generate, format, and automatically save structural plots to experiment-specific directories:

* **Training-Cost Curves:** Plots the binary cross-entropy loss against gradient descent iterations for both training and cross-validation sets to visually diagnose convergence speed, underfitting, or potential overfitting.
* **Confusion Matrix Heatmaps:** Transforms the raw numerical counts from `metrics.py` into a clean, color-mapped $2 \times 2$ grid complete with text annotations, axis normalization, and true-versus-predicted classification labels.

While I leveraged code-generation assistance to handle the tedious Matplotlib styling parameters (such as color maps, font sizing, and tick alignments), I structurally designed the data pipelines feeding these plots and interpret their visual outputs as part of the core experimentation process.

---

### Experiment-specific data files

The files:

* `experiment_1_data.py`
* `experiment_2_data.py`

contain the dataset preparation required for each binary classification task.

By separating the data logic, I prevented the training scripts from becoming unnecessarily crowded and made the differences between the two experiments easier to understand.

---

### Training scripts

The files:

* `train_experiment_1.py`
* `train_experiment_2.py`

control the complete training and evaluation workflows for their respective experiments.

Through these files, I connect:

* data preparation
* preprocessing
* model training
* hyperparameter configuration
* predictions
* metrics
* plots
* final outputs

---

## 6. General Machine-Learning Workflow

I followed the same overall pipeline for both experiments:

```text
Load Iris dataset
        ↓
Prepare experiment-specific binary targets
        ↓
Create training, cross-validation, and test sets
        ↓
Calculate normalization statistics from training data
        ↓
Normalize all three subsets
        ↓
Initialize weights and bias
        ↓
Train using gradient descent
        ↓
Track binary cross-entropy cost
        ↓
Compare hyperparameters using cross-validation data
        ↓
Select the final configuration
        ↓
Evaluate final predictions
        ↓
Generate confusion matrices and cost plots
        ↓
Perform error analysis
        ↓
Document the experiment
```

Through this structure, I separated:

* model training
* model selection
* final evaluation

I used the cross-validation set for hyperparameter experimentation, while the test set was intended to represent unseen data.

---

## 7. Experiment 1: Setosa vs Non-Setosa

### Problem definition

For Experiment 1, I converted the three-class Iris dataset into:

* **Class 1:** Setosa
* **Class 0:** Non-Setosa

The dataset contained:

* 50 positive examples
* 100 negative examples
* 150 total examples

The split produced:

* 105 training examples
* 22 cross-validation examples
* 23 test examples

### Final configuration

| Parameter                   | Value |
| --------------------------- | ----- |
| Learning rate               | 1.0   |
| Gradient-descent iterations | 1,000 |
| Classification threshold    | 0.5   |

### Cost results

| Cost                        | Value  |
| --------------------------- | ------ |
| Initial training cost       | 0.6931 |
| Final training cost         | 0.0011 |
| Final cross-validation cost | 0.0087 |
| Final test cost             | 0.0006 |

My model achieved perfect classification metrics on the training, cross-validation, and test subsets.

However, I did not interpret this result as evidence of a highly advanced classifier.

Setosa is strongly separable from Versicolor and Virginica, especially using petal measurements. Therefore, I primarily used Experiment 1 as:

* a pipeline verification experiment
* a test of my manual logistic regression implementation
* an investigation of convergence
* an investigation of learning-rate behaviour
* an investigation of numerical saturation

Generated plots:

* `plots/experiment_1/training_cost_curve.png`
* `plots/experiment_1/crossValidation_confusionMatrix.png`
* `plots/experiment_1/test_confusionMatrix.png`

---

## 8. Experiment 2: Versicolor vs Virginica

### Problem definition

For Experiment 2, I removed Setosa and classified:

* **Class 0:** Versicolor
* **Class 1:** Virginica

The dataset contained:

* 50 Versicolor examples
* 50 Virginica examples
* 100 total examples

The split produced:

* 70 training examples
* 15 cross-validation examples
* 15 test examples

### Final configuration

| Parameter                   | Value |
| --------------------------- | ----- |
| Learning rate               | 0.03  |
| Gradient-descent iterations | 1,000 |
| Classification threshold    | 0.5   |

### Final classification results

| Dataset          | Accuracy | Precision | Recall  | F1      |
| ---------------- | -------- | --------- | ------- | ------- |
| Training         | 97.14%   | 97.14%    | 97.14%  | 97.14%  |
| Cross-validation | 100.00%  | 100.00%   | 100.00% | 100.00% |
| Test             | 86.67%   | 85.71%    | 85.71%  | 85.71%  |

Experiment 2 created a more meaningful classification problem because Versicolor and Virginica measurements overlap.

My model made two test errors:

* one false positive
* one false negative

I analysed both examples using:

* original feature values
* normalized feature values
* predicted probabilities
* classification threshold
* individual feature contributions

Generated plots:

* `plots/experiment_2/training_cost_curve.png`
* `plots/experiment_2/crossValidation_confusionMatrix.png`
* `plots/experiment_2/test_confusionMatrix.png`

---

## 9. Comparison of the Two Experiments

| Aspect               | Experiment 1                        | Experiment 2                                  |
| -------------------- | ----------------------------------- | --------------------------------------------- |
| Task                 | Setosa vs Non-Setosa                | Versicolor vs Virginica                       |
| Total examples       | 150                                 | 100                                           |
| Class difficulty     | Relatively easy                     | More difficult                                |
| Class overlap        | Low                                 | Higher                                        |
| Main purpose         | Implementation verification         | Hyperparameter tuning and error analysis      |
| Final learning rate  | 1.0                                 | 0.03                                          |
| Final iterations     | 1,000                               | 1,000                                         |
| Test behaviour       | Perfect classification              | Two classification errors                     |
| Most useful analysis | Convergence and numerical behaviour | Misclassification and generalization analysis |

I did not build the two experiments as separate implementations.

I reused the same:

* model functions
* metric functions
* normalization logic
* training structure
* visualization utilities

The main differences were:

* class preparation
* dataset size
* class overlap
* selected hyperparameters
* resulting model behaviour

I used Experiment 1 to verify that my implementation worked correctly on an easily separable problem.

I used Experiment 2 to test the same implementation under a more challenging class boundary.

---

## 10. Hyperparameter Experimentation

I investigated the relationship between:

* learning rate
* number of gradient-descent iterations
* convergence speed
* training cost
* cross-validation cost
* classification metrics

One important observation was that different learning-rate and iteration combinations could reach almost the same solution.

For example, in Experiment 2:

```text
learning rate = 0.03, iterations = 1,000
```

and:

```text
learning rate = 0.01, iterations = 3,000
```

produced nearly identical costs and identical predictions.

I selected the configuration using a learning rate of 0.03 because it reached the same predictive result with fewer gradient-descent updates.

I also observed that the configuration with the lowest binary cross-entropy did not always produce the strongest thresholded classification metrics.

Binary cross-entropy evaluates predicted probabilities, while accuracy and related metrics evaluate classes after applying the 0.5 threshold.

---

## 11. Debugging and Development Lessons

I encountered several real debugging and implementation challenges while developing the project.

### Label validation

During Experiment 2, I created the binary target correctly, but my initial splitting code accidentally used the old labels 1 and 2 instead of the converted labels 0 and 1.

I detected the problem by checking the unique target values.

I discarded the invalid outputs and repeated the experiment using the correct binary targets.

### Loop-based implementation

My first implementations relied heavily on Python loops.

This was useful for understanding the logic line by line, but it also demonstrated why NumPy operations are preferred for numerical computing.

In a future version, I could replace more loop-based calculations with vectorized NumPy expressions.

### Array-shape handling

My implementation required careful handling of:

* feature-matrix shape
* weight-vector shape
* target-vector shape
* gradient-vector shape
* scalar bias values
* dot-product direction

Understanding these shapes was an important part of converting the mathematical equations into functioning code.

### Numerical stability

I observed that very confident sigmoid probabilities could approach exactly 0 or 1 because of floating-point limitations.

I added probability clipping before logarithmic operations to keep binary cross-entropy mathematically valid and numerically stable.

---

## 12. Project Development Approach

I developed the project incrementally.

My initial plan established the basic architecture, but I added several components as the practical requirements became clearer, including:

* separate experiment data files
* reusable visualization functions
* experiment-specific documentation
* cost-curve plotting
* confusion-matrix plotting
* individual error analysis

This combination of initial planning and controlled iteration worked effectively, although a more detailed architecture could reduce restructuring in a future project.

I committed and pushed the repository to GitHub regularly throughout development.

This created a record of:

* model development
* debugging
* experiment progression
* documentation updates
* visualization additions

---

## 13. Authorship and Use of Assistance

I wrote and understood the core machine-learning logic myself.

This includes:

* preprocessing logic
* sigmoid
* cost calculation
* gradient calculation
* gradient descent
* prediction logic
* evaluation metrics
* experiment execution
* hyperparameter experimentation
* result interpretation
* debugging

I used generated-code assistance for implementing the Matplotlib visualization utilities in `visualization.py`.

However, I decided the purpose and requirements of the plots and handled their integration, generated outputs, usage, and interpretation.

I did not create this project through blind code generation. I coded the core functions myself, understand the implemented model, and can explain its complete flow.

---

## 14. Main Learning Outcomes

This project significantly improved my understanding of:

* the complete ML development flow
* translating mathematical equations into Python
* practical logistic regression implementation
* gradient-descent behaviour
* learning-rate selection
* iteration-count selection
* training, validation, and test separation
* probability loss versus classification accuracy
* hyperparameter tuning
* NumPy array shapes
* manual metric calculation
* project file organization
* visualization integration
* debugging ML pipelines
* technical documentation
* Git and GitHub workflow

The project also taught me that implementing the model is only one part of machine learning.

A complete project also requires:

* validation
* experimentation
* interpretation
* error analysis
* reproducibility
* documentation

The most satisfying part of the project was completing a detailed implementation while understanding the complete logic instead of treating the code as a black box.

---

## 15. Project Limitations

My project has several limitations:

* the Iris dataset is small and clean
* both tasks use binary classification
* the model learns only a linear decision boundary
* evaluation depends on one random split
* the validation and test subsets are small
* I did not implement regularization
* I did not perform k-fold cross-validation
* the classification threshold remained fixed at 0.5
* the implementation contains loop-based calculations that could be vectorized
* I viewed the Experiment 2 test set for two finalist configurations
* I have not yet included a framework comparison

These limitations are appropriate for my first from-scratch ML project, but I acknowledge them clearly.

---

## 16. Future Improvements

Possible extensions include:

### Model comparison

I could implement the same tasks using sklearn logistic regression and compare:

* learned parameters
* predictions
* accuracy
* binary cross-entropy
* execution time

### Regularization

I could add L2-regularized logistic regression and study its effect on:

* parameter magnitude
* cost
* validation performance
* overfitting

### Multiclass classification

I could extend the implementation from binary logistic regression to multiclass softmax regression using all three Iris species.

### Improved validation

I could use:

* stratified k-fold cross-validation
* repeated random splits
* multiple random seeds
* mean and standard deviation of evaluation metrics

### Improved visualization

I could add:

* class scatter plots
* decision-boundary visualization
* learning-rate comparison curves
* feature-distribution plots

### Threshold tuning

I could investigate thresholds other than 0.5 and study the resulting precision-recall trade-off.

### Larger dataset

I could apply the implementation to a larger and noisier binary classification dataset.

### Vectorization

I could replace more Python loops with NumPy vectorized operations and compare:

* readability
* execution time
* output consistency

---

## 17. Final Conclusion

In this project, I successfully implemented and evaluated binary logistic regression from scratch.

I went beyond calling a framework training function by manually implementing the complete learning and evaluation process.

I used two related experiments to test the same model under different levels of difficulty:

* Setosa vs Non-Setosa verified my implementation on a strongly separable task.
* Versicolor vs Virginica provided more meaningful tuning, misclassification, and generalization analysis.

Through this project, I demonstrated:

* an understanding of logistic regression
* the ability to translate ML mathematics into code
* practical Python and NumPy skills
* structured project organization
* hyperparameter experimentation
* manual model evaluation
* numerical debugging
* visual analysis
* technical documentation
* GitHub development discipline

I consider this a detailed foundational machine-learning implementation rather than an advanced production system.

The project's main value is that I understand the complete model, experimentation process, results, limitations, and errors instead of hiding them behind a ready-made `.fit()` call.