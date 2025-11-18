### 1. MLPClassifier (Multi-layer Perceptron)
Framework: scikit-learn

#### Architecture:

Input Layer: Automatically inferred from feature dimension

Hidden Layers:

First Hidden Layer: 256 neurons

Second Hidden Layer: 128 neurons

Activation Function: ReLU

Output Layer: Softmax (internally for multi-class)

Solver (Optimizer): Adam

Learning Rate: 0.0122

L2 Regularization (alpha): 1.55e-5

Epochs (max_iter): 587

Random State: 42

### 2. RandomForestClassifier
Framework: scikit-learn

#### Architecture:

Model Type: Ensemble of decision trees (Bagging method)

Number of Trees: 100

Max Tree Depth: 15

Minimum Samples to Split a Node: 2

Minimum Samples per Leaf: 1

Feature Selection per Split: sqrt (i.e., √n_features)

Bootstrap Sampling: Enabled

Random State: 42

### 3. LogisticRegression
Framework: scikit-learn

#### Architecture:

Model Type: Linear classifier

Multi-class Strategy: One-vs-Rest (multi_class='auto')

Solver: L-BFGS (suitable for multiclass, smooth convex optimization)

Max Iterations: 1000

Regularization: L2 by default (can be tuned)

### 4. CatBoostClassifier
Framework: CatBoost

#### Architecture:

Model Type: Gradient Boosting Decision Trees (GBDT)

Iterations (Trees): 250

Eval Metric: Accuracy

Verbose: Logs metrics every 50 iterations

Auto Handles: Categorical Features, Missing Values

CatBoost builds oblivious trees, meaning each split at the same depth uses the same condition — useful for fast inference and consistent training.

### 5. XGBClassifier
Framework: XGBoost

#### Architecture:

Model Type: Gradient Boosted Decision Trees

Booster: dart (Dropouts meet Multiple Additive Regression Trees)

Number of Estimators: 1000 trees

Max Tree Depth: 10

Learning Rate (eta): 0.7 (aggressive)

Subsample (rows): 80%

Colsample_bytree (features): 80%

Regularization:

reg_lambda (L2 penalty): 18

reg_alpha (L1 penalty): 0.6

Eval Metric: Log loss

### 6. ExtraTreesClassifier
Framework: scikit-learn

#### Architecture:

Model Type: Ensemble of Extremely Randomized Trees (Extra Trees)

Number of Trees: 250

Tree Structure: Similar to Random Forest but splits are chosen at random rather than selecting the best threshold.

Bootstrap Sampling: Disabled by default (uses the whole dataset for each tree)

Random State: 42

Key Characteristics:

More randomness than Random Forest → Lower variance

Usually faster than RandomForest due to randomized thresholds

Good for reducing overfitting when features are noisy

### 7. SVC (Support Vector Classifier)
Framework: scikit-learn

#### Architecture:

Model Type: Support Vector Machine for Classification

Kernel: RBF (Radial Basis Function → maps input to higher dimensions)

Probability Estimates: Enabled via probability=True (uses Platt scaling, increases training time)

Regularization (C): Default = 1.0 (can be tuned)

Gamma: Default = ‘scale’ (1 / (n_features * X.var()))

