---
title: 'Safetycage: A Misclassification Detection Toolkit'
tags:
  - Python
  - Machine learning
  - Deep learning
  - Misclassification Detection
  - AI
authors:
  - given-names: Joel Finn
    surname: Bjervig
    orcid: 0009-0003-8579-1162
    equal-contrib: true
    affiliation: 1
  - given-names: Julia
    surname: Qiu
    equal-contrib: true
    affiliation: 2

affiliations:
 - name: SINTEF AS, Norway
   index: 1
 - name: Waterloo University, Canada
   index: 2

date: 06 March 2026
bibliography: paper.bib
---

# Summary

`Safetycage` is a Python package that provides a unified, framework-agnostic interface for misclassification detection in machine learning classifiers. It implements several detection methods — including Maximum Softmax Probability (MSP) [@HendrycksG16c], DOCTOR [@Granese21], Mahalanobis distance [@pmlr-v233-johnsen24a], and SPARDACUS [@johnsen2024spardacus] under a common abstraction that decouples the detection algorithm from the user's choice of model and data. Given a trained classifier, `Safetycage` learns to identify individual samples that are likely to be misclassified, enabling practitioners to flag unreliable predictions before they are acted upon.

# Statement of Need

Deployed classifiers inevitably make errors, yet most prediction pipelines provide no mechanism to flag individual predictions as untrustworthy. Misclassification detection methods (MDMs) address this gap by assigning a per-sample score that indicates the likelihood of an error, but implementations are typically scattered across framework-specific research code with incompatible interfaces.

`Safetycage` brings multiple MDMs under a single, class-based API. Users wrap their own model and data in lightweight modules (`ModelModule` and `DataModule`), after which any detection method can be trained, evaluated, and compared on the same data with minimal code changes. The current implementation focuses on classification tasks and supports any model that exposes predictions or intermediate activations, regardless of the underlying framework (scikit-learn, PyTorch, TensorFlow, etc.).

# State of the Field

Several software packages address related aspects of machine learning model reliability. MAPIE [@Cordier_Flexible_and_Systematic_2023] provides model-agnostic prediction intervals and prediction sets using conformal prediction techniques, but this is not used further down in the pipeline to flag misclassifications. Cleanlab [@cleanlab] focuses on identifying label errors in training data using confident learning, rather than detecting misclassifications at inference time. PyTorch-OOD [@pytorch_ood] provides Out-of-Distribution detection methods, but is restricted to the PyTorch ecosystem. Deepchecks [@Chorev_Deepchecks_A_Library_2022] and Microsoft's Responsible AI Toolbox [@microsoft_raitoolbox] offer comprehensive model auditing suites, but are designed for batch evaluation rather than providing an extensible API for implementing and comparing detection algorithms. NetCal [@Kueppers_2020_CVPR_Workshops] and Scores [@Leeuwenburg2024] focus on calibrating probability outputs, which is a form of error mitigation rather than explicit identification of misclassified samples.

`Safetycage` bridges the gap between prediction interval and specialized OOD detection tools on one side, and general-purpose auditing frameworks on the other side by providing a unified, framework-agnostic abstraction layer specifically for misclassification detection.


# Software Design

`Safetycage` is built around three abstract base classes that enforce a modular, framework-agnostic structure through composition. \autoref{fig:architecture}


![Caption text.\label{fig:architecture}](figure.png)


**`DataModule`** defines how data is loaded, transformed, and partitioned. Subclasses implement `_load_data()` to ingest data from any source, `_transform()` to apply preprocessing, and `_split()` to divide data into training and validation sets. The module also exposes dataset properties such as `num_classes` and `classes`, which are consumed by the detection methods.

**`ModelModule`** wraps the user's trained classifier behind a uniform interface. Subclasses implement `_get_predictions()` to obtain class outputs and `_get_activations()` to extract intermediate layer representations. Because the interface depends only on NumPy arrays, any model — whether built with scikit-learn, PyTorch, TensorFlow, or another framework — can be integrated by implementing these methods.

**`SafetyCage`** is the base class for detection methods. It composes a `ModelModule` and a `DataModule`, and delegates framework-specific concerns to them. Subclasses implement `train_cage()` to learn detection parameters from calibration data and `predict()` to compute per-sample scores indicating misclassification risk. The base class provides shared functionality: `flag()` applies a decision threshold to the scores, `find_best_threshold()` selects the optimal threshold by maximizing a user-chosen metric, and `save_cage()` / `load_cage()` handle serialization of trained parameters.

This design allows any combination of detection method, model, and dataset to be composed freely. Users can compare multiple methods on the same data and model, or apply the same method across different models, with no code duplication.

# Implemented Methods

`Safetycage` currently implements four detection methods. Each computes a per-sample score which indicates misclassification risk. An optimal decision threshold is selected on calibration data - disjont from data used to construct ML model - by maximizing a user-chosen metric. For full mathematical details of each method, we refer to the original publications.

| Method | Approach |
|--------|----------|
| MSP [@HendrycksG16c] | Thresholds on the maximum softmax probability of the classifier | 
| DOCTOR [@Granese21]| Estimates a rejection rule from the ratio of correct/incorrect class distributions | 
| Mahalanobis [@pmlr-v233-johnsen24a] | Measures distance of layer activations from class-conditional Gaussian statistics | 
| SPARDACUS [@johnsen2024spardacus]| Projects activations to maximize separation between correct and incorrect distributions | 

# Examples of Use

We demonstrate the core `Safetycage` workflow using the Iris classification dataset and the Maximum Softmax Probability (MSP) method. The package is installed via PyPI:

```
pip install safetycage
```

The user first wraps their dataset and pre-trained classifier in the corresponding modules. The `DataModule` defines how data is loaded and split, while the `ModelModule` exposes prediction access:

```python
from safetycage.data import IrisDataModule
from safetycage.model import IrisModelModule
import joblib

data_module = IrisDataModule()
model = joblib.load("random_forest_model.joblib")
model_module = IrisModelModule(model)
```

A `SafetyCage` method is then instantiated with these modules and trained on validation data to learn the score distributions of correctly and incorrectly classified samples:

```python
from safetycage import MSP

cage = MSP(model_module=model_module, data_module=data_module)
cage.train_cage()
```

Finally, the optimal detection threshold is selected by maximizing a chosen metric (here, the Matthews Correlation Coefficient) over candidate thresholds. The trained detector can then flag likely misclassifications on new data:

```python
alpha, mcc = cage.find_best_threshold_flag(metric="mcc")
flags = cage.flag(alpha=alpha)
```

The returned `flags` array indicates, for each test sample, whether the detector considers it likely to be misclassified. Full worked tutorials covering MSP, SPARDACUS, and other methods on various datasets are available in the [`examples`](https://github.com/SINTEF/safetycage/tree/main/examples) directory of the repository and in the [package documentation](https://safetycage.readthedocs.io/).

# Future Work

Future development includes extending `Safetycage` to regression tasks, where a misdetection is defined as a prediction that deviates from the ground truth by more than a threshold distance $d$.

# AI Usage Disclosure

Generative AI tools were used to create some of the docstrings and documentation of this software. All generated material has been verified by the authors.

# Acknowledgements

This work was supported by the Research Council of Norway through projects EXAIGON (grant no. XXX) and THEMIS (grant no. XXX), and by SINTEF internal funding.

We acknowledge contributions from Pål Johnsen and Filippo Remonato for developing two new misclassification methods, namely Mahalanobis and SPARDACUS.

# References
