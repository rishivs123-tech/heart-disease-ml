# Chapter 2: Literature Review

## 2.1 Clinical Diagnostic Foundations of Heart Disease
Coronary artery disease (CAD) develops through the progressive accumulation of atherosclerotic plaques within the coronary arteries, restricting oxygenated blood delivery to the myocardium. Angiography represents the definitive gold standard for CAD diagnosis, defining clinically significant disease as $\ge 50\%$ diameter stenosis in at least one major epicardial coronary vessel. However, angiography is an invasive, costly, and resource-intensive procedure carrying procedural risks.

Consequently, non-invasive risk stratification relies on routine demographic and diagnostic biomarkers:
- **Demographic Attributes**: Patient age and sex (with epidemiological studies consistently showing higher early incidence in males).
- **Symptom Characterization**: Chest pain type (`cp`), categorized into typical angina, atypical angina, non-anginal pain, and asymptomatic presentation.
- **Hemodynamic Parameters**: Resting arterial blood pressure (`trestbps`) and maximum heart rate achieved during peak physical exertion (`thalch`).
- **Serum Lipid Profiles**: Fasting blood glucose (`fbs`) and total serum cholesterol (`chol`).
- **Electrocardiography & Exercise Stress Tests**: Resting ECG abnormalities (`restecg`), ST segment depression induced by exercise relative to rest (`oldpeak`), and the slope of the peak exercise ST segment (`slope`).
- **Advanced Specialized Scintigraphy**: Fluoroscopic visualization of major calcified vessels (`ca`) and radioactive thallium myocardial perfusion scintigraphy (`thal`).

## 2.2 Machine Learning in Tabular Healthcare Analytics
Supervised classification algorithms have been extensively investigated for cardiac disease risk estimation:
1. **Logistic Regression (LR)**: Represents the standard classical generalized linear model. While providing high intrinsic interpretability through monotonic log-odds coefficients, standard logistic regression struggles to model non-linear boundaries without manual interaction feature engineering.
2. **Support Vector Machines (SVM)**: Employs the kernel trick (specifically the Radial Basis Function kernel) to project clinical attributes into infinite-dimensional Hilbert spaces, constructing optimal separating hyperplanes with maximum margin. While robust against high-dimensional collinearity, SVM decision boundaries lack intuitive transparency.
3. **K-Nearest Neighbors (KNN)**: A non-parametric instance-based algorithm classifying patients based on Euclidean or Minkowski distance in feature space. KNN exhibits extreme vulnerability to feature scaling disparities and missing coordinates.
4. **Random Forest (RF)**: An ensemble bagging technique constructing decorrelated decision trees using bootstrap aggregating (bagging) and random feature subspace splitting. RF substantially reduces variance and resists overfitting on small tabular cohorts.
5. **Gradient Boosting (GB)**: A sequential boosting framework that iteratively fits weak decision trees to the negative gradients of the binomial loss function. Gradient Boosting typically demonstrates superior predictive performance on tabular benchmarks.

## 2.3 Methodologies for Incomplete Clinical Data
Missing data in biomedical datasets generally falls into three theoretical categories defined by Rubin:
- **Missing Completely at Random (MCAR)**: Missingness probability is entirely independent of observed and unobserved variables.
- **Missing at Random (MAR)**: Missingness depends on observed features but not on the unobserved value itself.
- **Missing Not at Random (MNAR)**: Missingness is directly related to the value of the missing measurement itself (e.g., severely ill patients unable to undergo exercise stress testing).

Handling paradigms range from naive single-value substitution (median/mode) to advanced multivariate algorithms:
- **Missingness Indicators (M2)**: Augmenting feature vectors with binary indicator columns denoting whether an attribute was originally missing, allowing algorithms to learn informative missingness patterns.
- **K-Nearest Neighbors Imputation (M3)**: Imputing missing values by averaging the non-missing coordinates of the $k$ most similar patient instances in normalized continuous space.
- **Multivariate Imputation by Chained Equations (M4 / MICE)**: Models each variable with missing values as a function of all other covariates in an iterative Gibbs-sampling scheme, preserving multivariate covariance structures.

## 2.4 Explainable AI (XAI) and SHAP Formulations
To overcome the opacity of complex algorithms, Explainable AI frameworks have evolved rapidly:
- **LIME (Local Interpretable Model-agnostic Explanations)** fits sparse linear models to local perturbations around a given instance. However, LIME suffers from sampling instability and heuristic kernel parameters.
- **SHAP (SHapley Additive exPlanations)** establishes a unified framework grounded in cooperative game theory. For an input vector $x$ and model $f$, the Shapley value $\phi_i$ of feature $i$ represents its average marginal contribution across all possible feature coalitions $S \subseteq F \setminus \{i\}$:
$$\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} [f(S \cup \{i\}) - f(S)]$$

SHAP satisfies four desirable theoretical axioms: Efficiency (local accuracy), Symmetry, Dummy (null player), and Additivity. Model-specific implementations—notably **TreeSHAP** for decision tree ensembles and **LinearSHAP** for generalized linear models—enable polynomial-time exact computation.

## 2.5 The Explanation Stability Research Gap
Recent theoretical inquiries in machine learning safety have highlighted that predictive performance does not imply attribution stability. When a model is retrained on alternate bootstrap resamples or subjected to alternative imputation strategies:
- The global importance ordering of clinical features may fluctuate substantially.
- Individual patient attributions may invert polarity (e.g., an attribute switching from risk-increasing to protective).
This dissertation directly addresses this gap by formalizing quantitative stability metrics (Spearman rank correlation, Top-$k$ Jaccard similarity, and patient-level Mean Absolute Difference) to evaluate the robustness of SHAP explanations.
