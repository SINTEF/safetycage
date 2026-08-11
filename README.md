<p align="center">
  <img src="docs/_static/logo-2.png" height="300" alt="Project Logo">
</p>

SafetyCage is a Python package for **detecting misclassified predictions** from machine learning models in classification tasks. It provides a unified interface for multiple statistical detection methods, enabling users to quantify prediction reliability and flag potentially incorrect outputs across different models and datasets easily.

It is available on PyPI here: https://pypi.org/project/safetycage/.

## Background

The idea behind safetycage is that we can find statistics on each predicted sample and compare that statistic to some statistic threshold to predict whether the sample prediction was incorrectly classified.

## Description
Machine learning models can produce incorrect predictions with high confidence. SafetyCage addresses this by providing post-hoc misclassification detection methods that operate on model outputs or internal representations.

The package includes several methods:

- MSP (Maximum Softmax Probability)
- DOCTOR (Error probability estimation)
- Mahalanobis (Distance-based statistical testing)
- SPARDACUS (Projection + density estimation approach)

Each method outputs a statistic or p-value that reflects how likely a prediction is to be incorrect.

Alternatively, you can implement your own method by initializing a base class from the safetycage abstract base class, which defines how methods should be implemented.

## Requirements
Currently, safetyCage **requires** Python 3.11.7. Consider creating an environment for your project with safetycage that uses Python 3.11.7.

> We are working to make this more flexible.

## Installation
Install via pip using the command:

```
pip install safetycage
```

The dependencies should be automatically installed when you install safetycage. If not, consider seeing the dependencies listed in the pyproject.toml available on [GitHub](https://github.com/SINTEF/safetycage/blob/main/pyproject.toml).


<!-- ## Visuals
- could put a gif on installing and using the package - don't think this is very necessary, should be obvious -->

## Tutorials & Examples

To learn how to use safetycage, check out the safetycage tutorials available at https://github.com/safety-cage/safetycage-tutorials. These provide full examples and tutorials on how to use safetycage, and thus also providing scripts to train models to test the safetycage methods on!

## Changelog
See the [CHANGELOG.MD](https://github.com/SINTEF/safetycage/blob/main/CHANGELOG.md) for details on versioning.

## Support
If you encounter issues or have questions:
- Open an issue on the repository: https://github.com/SINTEF/safetycage/issues.
- Check the [safetycage-tutorials repo](https://github.com/safety-cage/safetycage-tutorials) for examples.

<!-- ## Roadmap
- TBD -->

## Contributing
If you would like to contribute, please reach out to our safetycage team, listed below!

## Authors
*   **Pål Vegard Bun Johnsen** - [palVJ](https://github.com/palVJ)
*   **Joel Bjervig** - [stjoel](https://github.com/stjoel)
*   **Julia Qiu** - [jq11](https://github.com/jq-11)

## Acknowledgment

The MSP method was introduced by Hendrycks and Gimpel in *A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks*. 

The DOCTOR method was introduced by Granese et al. in *DOCTOR: A Simple Method for Detecting Misclassification Errors.* 

A proper citation for these methods is provided in the docstring of the code using these methods.

A special thank you goes to previous co-authors of the methods we have built, Filippo Remonato, Shawn Benedict, and Albert Ndur-Osei.

<!-- unsure how to cite these people -->

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Project status
Active and under development!

## Citation
If you use safetycage, please cite us!

<!-- Include bibtex/apa citation -->