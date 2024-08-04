# Movie Rating Prediction Project

## Introduction
This project aims to predict IMDb movie ratings using machine learning models. By exploring various algorithms, such as Support Vector Machines, Random Forest, Gradient Boosting, AdaBoost, and Ensemble methods, we strive to identify the most effective approach for this classification task.

## Methodology

### Data Pre-processing
1. **Data Cleaning**: Removed string columns as many models cannot interpret them.
2. **Handling Missing Values**: Imputed missing values using median values.
3. **Normalization**: Scaled numerical features to a range between 0 and 1 using MinMaxScaler.

### Model Selection and Training
The following models were trained and evaluated:
- **Support Vector Machine (SVM)**: Effective in high-dimensional spaces and for cases with dimensions exceeding samples.
- **Random Forest Classifier**: Constructs multiple decision trees and uses the mode for classification, known for high accuracy.
- **Gradient Boosting Classifier**: Builds an additive model in stages, optimizing a loss function.
- **AdaBoost Classifier**: Combines multiple weak classifiers, adapting to difficult instances.
- **Voting Classifier**: Combines predictions from multiple models (RF, Gradient Boosting, SVM) based on majority voting.

### Evaluation
The performance of each model was evaluated using accuracy as the primary metric. Cross-validation was employed to ensure robustness. The dataset was split into training (70%) and validation (30%) sets.

## Results

### Performance Comparison
The performance of the classifiers was evaluated using the following metrics:
- **Train Accuracy**: Accuracy on the training set.
- **Validate Accuracy**: Accuracy on the validation set.
- **Overall Accuracy**: Accuracy on the entire dataset.

| Model                | Train Accuracy | Validate Accuracy | Overall Accuracy |
|----------------------|----------------|-------------------|------------------|
| SVM                  | 0.690771       | 0.691796          | 0.691079         |
| Random Forest        | 0.997146       | 0.710643          | 0.911119         |
| Gradient Boosting    | 0.813035       | 0.721729          | 0.785619         |
| AdaBoost Classifier  | 1.000000       | 0.696231          | 0.908788         |
| Ensemble Classifier  | 0.895814       | 0.711752          | 0.840546         |

### Cross-Validation Results
| Model               | Mean Accuracy | Standard Deviation |
|---------------------|---------------|--------------------|
| SVM                 | 0.6703        | 0.0222             |
| Random Forest       | 0.6822        | 0.0221             |
| Gradient Boosting   | 0.6960        | 0.0268             |
| AdaBoost Classifier | 0.6827        | 0.0225             |
| Ensemble Classifier | 0.6808        | 0.0249             |

## Discussion and Analysis

### Imbalanced Distribution
The dataset showed an imbalanced distribution of IMDb scores, with most movies clustered in ratings 2 and 3. To address this, class weight balancing was applied to the RF and SVM models, leading to improved accuracy for RF but a significant drop for SVM.

### Correlation and Feature Extraction
The correlation matrix revealed strong relationships among features, indicating potential multicollinearity. Tree-based models handled this well, while SVM models showed minimal performance differences after removing multicollinear variables.

### Outliers and Overfitting
Outliers were identified and filtered using z-scores. However, this led to overfitting in the RF and AdaBoost models. Balancing data cleaning with preserving variability is crucial to ensure models generalize effectively.

### Model Performance
- **SVM**: Effective but sensitive to imbalanced datasets.
- **Random Forest**: High accuracy but prone to overfitting.
- **Gradient Boosting**: Balanced performance but requires careful tuning.
- **AdaBoost**: Overfits training data, resulting in poor generalization.
- **Ensemble**: Best overall performance, leveraging the strengths of individual models.

## Conclusion
The Ensemble model emerged as the top performer, highlighting the importance of combining individual model strengths for robust predictions. Addressing imbalanced datasets, feature correlation, and overfitting is crucial for accurate movie rating prediction.

## Visualizations
The following visualizations were created:
- IMDb score distribution
- Feature correlation matrix
- Box plots for numeric features
- Confusion matrices for each model

These visualizations provided insights into data characteristics and model performance, guiding the selection and evaluation of machine learning models.

#### Reference

This project is part of COMP30027 Machine Learning Semester 1 2024 course of the University of Melbourne.