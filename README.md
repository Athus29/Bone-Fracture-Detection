# Diagnostic Support System for Fracture Detection Using Deep Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-2.x-red)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-0.24-yellow)

## Overview

This project focuses on developing a **Diagnostic Support System** for the automatic detection of bone fractures from X-ray images using **deep learning** and **transfer learning** techniques. The system leverages state-of-the-art architectures such as **ResNet** and **DenseNet** to classify X-ray images into fractured and non-fractured categories. By improving diagnostic accuracy and efficiency, this system has the potential to assist radiologists in making faster, more reliable diagnoses.

## Models Used

1. **ResNet (Residual Networks)**:
   - ResNet enables the training of very deep networks by incorporating skip connections, which solve the problem of vanishing gradients and allow for more effective feature learning.
   - While ResNet performed well in this project, it showed signs of overfitting, with a training accuracy of approximately 70% and a validation accuracy of about 40%.

2. **DenseNet (Densely Connected Convolutional Networks)**:
   - DenseNet makes use of dense connections between layers, improving feature reuse and reducing the number of parameters. This allows for efficient training while avoiding overfitting.
   - DenseNet outperformed ResNet with a training accuracy of 73% and a validation accuracy of 66%, making it the preferred model for this task.

## Required Libraries

To run this project, you will need the following Python libraries:

- **TensorFlow** (>= 2.x)
- **Keras** (>= 2.x)
- **Scikit-learn** (>= 0.24)
- **Matplotlib** (for plotting metrics)
- **NumPy**
- **Pandas**
- **OpenCV** (optional, for additional image preprocessing)

You can install the required libraries using the following command:

```bash
pip install tensorflow keras scikit-learn matplotlib numpy pandas opencv-python
```

## Performance Metrics

The following metrics were used to evaluate the performance of the models:

- **Accuracy**: Measures the overall correctness of the model by calculating the proportion of correctly classified instances.
- **Precision**: Focuses on the accuracy of positive predictions.
- **Recall**: Measures how well the model identifies true positive cases.
- **F1 Score**: The harmonic mean of precision and recall, providing a balance between the two.
- **Confusion Matrix**: Displays the number of true positives, true negatives, false positives, and false negatives, providing a clear picture of the model's performance.

### Results:

- **ResNet**:
  - Training Accuracy: 70%
  - Validation Accuracy: 40%
  - F1 Score for the Fractured class: 0.40

- **DenseNet**:
  - Training Accuracy: 73%
  - Validation Accuracy: 66%
  - F1 Score for the Fractured class: 0.83 (Precision: 0.73, Recall: 0.95)

DenseNet demonstrated better generalization and robustness, making it more suitable for clinical applications.

