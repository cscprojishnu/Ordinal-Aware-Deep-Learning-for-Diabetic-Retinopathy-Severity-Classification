# Ordinal-Aware Deep Learning for Diabetic Retinopathy Severity Classification

> A Semester 2 B.Tech research/engineering project exploring ordinal-aware deep learning for seven-stage diabetic retinopathy (DR) severity classification using retinal fundus images.

---

## About This Project

This project was developed during **Semester 2 of my B.Tech** as an early step into deep learning, medical image analysis, and research-oriented machine learning.

The objective was to explore whether incorporating the **ordinal structure of diabetic retinopathy severity** into the learning process could provide a more meaningful prediction framework than treating all disease stages as completely independent classes.

The project uses an **ImageNet-pretrained EfficientNet-B0** backbone combined with two prediction branches:

1. A conventional **7-class classification head**
2. A **6-threshold ordinal regression head**

The final model jointly learns categorical class identity and the ordered progression of disease severity.

This project should be viewed as a **learning and research-development milestone rather than a publication-ready study**. It was my stepping stone toward more rigorous research involving larger datasets, stronger experimental protocols, external validation, explainability, and clinically meaningful evaluation.

---

## Why Diabetic Retinopathy?

Diabetic retinopathy is a progressive retinal disease associated with diabetes and is commonly assessed using retinal fundus photographs.

A particularly interesting property of DR classification is that its severity labels are **ordinal**.

The classes used in this project are:

| Label | Severity |
|------:|----------|
| 0 | No DR signs |
| 1 | Mild NPDR |
| 2 | Moderate NPDR |
| 3 | Severe NPDR |
| 4 | Very Severe NPDR |
| 5 | PDR |
| 6 | Advanced PDR |

This means that predicting class `3` instead of class `4` is fundamentally different from predicting class `0` instead of class `6`.

A conventional multiclass classifier does not explicitly encode this relationship.

This motivated the use of an ordinal learning component.

---

## Model Architecture

The proposed architecture is based on:

**EfficientNet-B0 + Classification Head + Ordinal Regression Head**

```text
                    Retinal Fundus Image
                         384 × 384
                              |
                              v
                   ImageNet Pretrained
                      EfficientNet-B0
                              |
                              v
                     512-D Feature Vector
                              |
                 +------------+------------+
                 |                         |
                 v                         v
          Classification Head       Ordinal Head
                 |                         |
                 v                         v
          7 Class Outputs            6 Thresholds
                 |                         |
                 |              P(Y > 0), P(Y > 1), ...
                 |              P(Y > 5)
                 |                         |
                 +------------+------------+
                              |
                              v
                    Ordinal-Aware Prediction
