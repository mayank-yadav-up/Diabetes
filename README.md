
# Diabetes Prediction Project

Predict diabetes outcomes using a machine learning model built with Python and the Pima Indians Diabetes Dataset. This project employs a Random Forest Classifier to predict whether a patient has diabetes based on features like glucose levels, BMI, and age.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

The goal of this project is to build a predictive model for diabetes using the Random Forest Classifier. The process includes:

- Loading and exploring the dataset
- Visualizing key features to understand data distributions
- Preprocessing the data (handling missing values, scaling features)
- Training a Random Forest model
- Evaluating the model using metrics like accuracy, confusion matrix, and classification report

---

## Dataset

The dataset used is the **Pima Indians Diabetes Dataset** (`diabetes.csv`), which contains the following features:

| Feature                   | Description                                      |
|---------------------------|--------------------------------------------------|
| Pregnancies               | Number of times pregnant                         |
| Glucose                   | Plasma glucose concentration                     |
| BloodPressure             | Diastolic blood pressure (mm Hg)                 |
| SkinThickness             | Triceps skin fold thickness (mm)                 |
| Insulin                   | 2-Hour serum insulin (mu U/ml)                   |
| BMI                       | Body mass index (weight in kg/(height in m)^2)   |
| DiabetesPedigreeFunction  | Diabetes pedigree function                       |
| Age                       | Age (years)                                      |
| Outcome                   | Target variable (0 = non-diabetic, 1 = diabetic) |

The dataset contains **no missing values**.

---

## Requirements

To run this project, you need the following Python libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Install them via the requirements file.

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/diabetes-prediction.git
   cd diabetes-prediction
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure the `diabetes.csv` dataset is in the project directory.**

---

## Usage

- **Open the Jupyter Notebook:**
  ```bash
  jupyter notebook Project.ipynb
  ```
  Run the notebook cells sequentially to:
  - Load and explore the dataset
  - Visualize feature distributions
  - Preprocess the data
  - Train and evaluate the Random Forest model

- **Alternatively, run as a Python script:**
  ```bash
  jupyter nbconvert --to script Project.ipynb
  python Project.py
  ```

---

## Project Structure

```
diabetes-prediction/
├── diabetes.csv          # Dataset file
├── Project.ipynb         # Jupyter Notebook with the analysis and model
├── requirements.txt      # List of required Python libraries
└── README.md             # Project documentation (this file)
```

---

## Model Training and Evaluation

The notebook follows these steps:

1. **Data Loading:**  
   Load the dataset using `pandas.read_csv('diabetes.csv')`.

2. **Data Exploration:**  
   - Display the first few rows (`data.head()`)
   - Check for missing values (`data.isnull().sum()`)

3. **Data Visualization:**  
   - Visualize feature distributions using seaborn and matplotlib
   - Plot a confusion matrix to show model performance

4. **Preprocessing:**  
   - Scale features using `StandardScaler`
   - Split the dataset into training and testing sets using `train_test_split`

5. **Model Training:**  
   - Train a Random Forest Classifier on the preprocessed data

6. **Evaluation:**  
   - **Accuracy Score:** Proportion of correct predictions
   - **Confusion Matrix:** Visualized as a heatmap
   - **Classification Report:** Precision, recall, F1-score, and support for each class

---

## Results

**Key Results:**

- **Accuracy:** ~75% on the test set

**Classification Report:**
```
              precision    recall  f1-score   support
         0       0.81      0.80      0.81        99
         1       0.65      0.67      0.66        55
  accuracy                           0.75       154
 macro avg       0.73      0.74      0.73       154
weighted avg     0.76      0.75      0.75       154
```

- The confusion matrix heatmap visually represents the model's performance, showing true positives, true negatives, false positives, and false negatives.

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes and commit (`git commit -m "Add feature"`)
4. Push to the branch (`git push origin feature-branch`)
5. Open a pull request




