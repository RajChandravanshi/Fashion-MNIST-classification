DataSet Link = https://www.kaggle.com/datasets/zalando-research/fashionmnist?select=fashion-mnist_train.csv

# Fashion-MNIST Classification Report

> **Note**: This is a learning project aimed at exploring and comparing basic deep learning models using **Keras** and **PyTorch** for fashion item classification on the Fashion-MNIST dataset

---

## 1. Introduction

This project explores the use of fully connected neural networks to classify clothing items from the Fashion-MNIST dataset. Implemented in both **Keras** and **PyTorch**, this study compares the architectures, training processes, and final performance of each model on a shared dataset split and identical preprocessing pipeline.

The key objectives are:
- To develop and train ANN models using Keras and PyTorch
- To evaluate their performance across training, validation, and test sets
---

## 2. Dataset Overview

Fashion-MNIST is a benchmark dataset provided by Zalando, containing grayscale images of fashion products. It includes 60,000 training images and 10,000 test images. Each image is:
- 28×28 pixels
- Grayscale (1 channel)
- Belonging to one of 10 classes

### Class Labels:
| Label | Class       |
|-------|-------------|
| 0     | T-shirt/top |
| 1     | Trouser     |
| 2     | Pullover    |
| 3     | Dress       |
| 4     | Coat        |
| 5     | Sandal      |
| 6     | Shirt       |
| 7     | Sneaker     |
| 8     | Bag         |
| 9     | Ankle boot  |

---

## 3. Data Splitting

To ensure a fair and consistent comparison, the dataset was split as follows:

- **Training Set**: 58,200 samples  
- **Validation Set**: 900 samples  
- **Test Set**: 900 samples  

This split was manually created from the original 60,000 training images. The test set used here is a separate hold-out set and **not** the official 10,000-image Fashion-MNIST test set.

---


## 4. Model Architectures

Both Keras and PyTorch models used the same ANN structure with three hidden layers, each followed by batch normalization and dropout.

### Architecture Summary

**Input Layer**  
- 784 input features (flattened image)

**Hidden Layer 1**  
- 128 neurons  
- ReLU activation  
- He-normal kernel initializer (Keras only)  
- Batch normalization  
- Dropout rate: 0.2

**Hidden Layer 2**  
- 64 neurons  
- ReLU activation  
- Batch normalization  
- Dropout rate: 0.2

**Hidden Layer 3**  
- 32 neurons  
- ReLU activation  
- Batch normalization  
- Dropout rate: 0.2

**Output Layer**  
- 10 neurons  
- Softmax activation (Keras)  
- Raw logits (PyTorch)  

---

## 5. ⚙️ Training Configuration

| Parameter       | Value                        |
|-----------------|------------------------------|
| Optimizer       | Adam                         |
| Loss Function   | CrossEntropy (PyTorch) / CategoricalCrossentropy (Keras) |
| Batch Size      | 64                           |
| Epochs          | ~20–30 (early stopping used) |
| Learning Rate   | Default framework setting    |
| Regularization  | Dropout (0.2), BatchNorm     |

Early stopping and model checkpointing were used to prevent overfitting and preserve the best weights.

---

## 6. Evaluation Metrics

The primary metric used to evaluate model performance is **classification accuracy** across the training, validation, and test sets.

### Final Results

| Framework | Train Accuracy | Validation Accuracy | Test Accuracy |
|-----------|----------------|---------------------|---------------|
| Keras     | 91.69%         | 89.01%              | 89.18%        |
| PyTorch   | 82.16%         | 87.91%              | 87.09%        |

---

## 7. Analysis and Discussion

- **Keras** achieved higher training accuracy and slightly better validation/test results, suggesting more effective learning on the current configuration.
- **PyTorch** showed lower training accuracy but competitive test accuracy, indicating better regularization and generalization.
- Both models benefited from **batch normalization** and **dropout**, which helped mitigate overfitting.
- The small validation set (900 samples) may limit the reliability of early stopping decisions.
- Differences in performance may stem from internal defaults (e.g., weight initialization, optimizer settings).

---

## 8. Conclusion

This project demonstrates that both Keras and PyTorch can effectively train ANNs on the Fashion-MNIST dataset with high accuracy. While Keras slightly outperformed PyTorch in this setup, both models achieved **over 87% accuracy** on the test set, which is notable for a simple fully connected architecture.

---


