PIRVision Human Activity Detection
================================
This project involves the classification of PIR sensor data combined with contextual tabular features (e.g., temperature and time) to detect human activity or environmental conditions. The dataset contains time-series data from multiple PIR sensors (`PIR_1` to `PIR_55`) and additional features like temperature and time. The goal is to build robust machine learning models that can classify the data into three distinct labels:

- **Label 0**: Vacancy
- **Label 1**: Stationary human presence
- **Label 2**: Other activity/motion

The project follows a structured pipeline, including data preprocessing, feature engineering, exploratory data analysis (EDA), and the development of multiple deep learning models. Each model is evaluated using 5-fold cross-validation to ensure generalization.

---

## Project Summary Table

| **Step**                       | **Description**                                                                                     | **Key Outputs**                                                                 |
| ------------------------------ | --------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| **1. Data Loading**            | Loaded the PIR sensor dataset and performed initial inspection.                                     | Dataset loaded into a Pandas DataFrame.                                         |
| **2. EDA**                     | Explored the dataset for class imbalance, feature distributions, and outliers.                      | Boxplots, histograms, and descriptive statistics.                               |
| **3. Outlier Handling**        | Replaced outliers in `PIR_1` with the median value to ensure uniformity across PIR columns.         | Cleaned dataset with consistent PIR sensor distributions.                       |
| **4. Feature Engineering**     | Engineered cyclical time features (`Hour_sin`, `Hour_cos`, etc.) and normalized temperature values. | Enhanced dataset with additional features for temporal and contextual patterns. |
| **5. Feature Scaling**         | Applied `MinMaxScaler` to PIR columns and `StandardScaler` to temperature for model compatibility.  | Scaled dataset ready for training.                                              |
| **6. Model 1: LSTM**           | Built an LSTM-based model to capture temporal dependencies in PIR sensor data.                      | Achieved ~95% accuracy with strong generalization.                              |
| **7. Model 2: LSTM + Tabular** | Combined LSTM for PIR data with a dense block for tabular features.                                 | Improved accuracy (~98%) by leveraging both modalities.                         |
| **8. Model 3: CNN**            | Developed a 1D-CNN model to extract local temporal patterns from PIR data.                          | Achieved ~95% accuracy with computational efficiency.                           |
| **9. Model 4: CNN + Tabular**  | Combined CNN for PIR data with a dense block for tabular features.                                  | Achieved ~98-99% accuracy with excellent generalization.                           |
| **10. Evaluation**             | Evaluated all models using 5-fold cross-validation and analyzed class-wise performance.             | Metrics: Accuracy, Macro F1-Score, Confusion Matrix.                            |
| **11. Insights**               | Summarized key findings, including class-wise performance and feature importance.                   | Identified strong correlation between temperature and Label 2.                  |

---

## Key Takeaways

1. **Hybrid Models Perform Best**: Combining PIR time-series data with tabular features significantly improves classification accuracy.
2. **Class Imbalance Challenges**: Class 1 (Stationary human presence) is the most challenging to classify due to overlap with other classes.
3. **Feature Importance**: Temperature and cyclical time features play a crucial role in distinguishing between classes.
4. **Generalization**: All models demonstrate strong generalization, with minimal overfitting observed during training.
--------------------------
This project implements classification models (LSTM, CNN, and hybrid models) to detect human activity or environmental conditions.
Files Included (from zip):
--------------------------
- `README.md` -> This file
- `team_26.ipynb` -> Main Jupyter notebook containing all code
- `team_26_model1_checkpoint.pth` -> PyTorch model checkpoint (LSTM)
- `team_26_model2_checkpoint.pth` -> PyTorch model checkpoint (LSTM + Tabular)
- `team_26_model3_checkpoint.pth` -> PyTorch model checkpoint (CNN)
- `team_26_model4_checkpoint.pth` -> PyTorch model checkpoint (CNN + Tabular)
- `team_26-checkpoint.ipynb` -> Auto-generated Jupyter checkpoint file (can be ignored)
    Note: This is automatically created by Jupyter to save intermediate notebook states. You do not need to open or run this file.

Libraries Used
--------------
The following Python libraries are required:

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
import seaborn as sns

All libraries are available via pip or conda. Ensure your environment has PyTorch and Scikit-learn installed.

How to Run the Code (If want to run the whole code)
-------------------
Step 1: Set Up Environment
Use Google Colab or Anaconda Jupyter Notebook.

Step 2: Dataset Path
If using Google Colab, upload the CSV file to your Drive in DL_Project folder:
from google.colab import drive
drive.mount('/content/drive')
dataset_path = "/content/drive/MyDrive/DL_Project/pirvision_office_dataset1.csv"

If using Anaconda (local machine), set the dataset path accordingly:
dataset_path = "C:\\Users\\your_username\\Downloads\\pirvision_office_dataset1.csv"

Step 3: Load Dataset (First Cell)
df = pd.read_csv(dataset_path)
df.head()

Step 4: Run Each Code Cell in Order
Once the dataset is loaded, execute each notebook cell in sequence. The notebook covers:
1. Data preprocessing
2. Feature scaling
3. Model definition (LSTM/CNN/Hybrid)
4. Train/validation/test splits using Stratified K-Fold
5. Training and validation
6. Performance evaluation (accuracy, F1-score, confusion matrix)
7. Visualization of training/validation loss and accuracy

Model checkpoints will be saved automatically.
To change the filename for saved models, update the path in the evaluate_model() function of the respective models.

How to Run the Code (For evaluation purposes)
-------------------
Step 1: Prepare Scaled Training Data:
Follow the steps sequentially in the notebook until you obtain the scaled training data dataframe.

Step 2: Evaluate the Model:
Once the data is prepared, run the evaluate_model() function at the end of the notebook.

Step 3: Run the Architecture Code for the Chosen Model:
For each model, navigate to the appropriate section in the notebook where the architecture code is defined and execute it.

Step 4: Execute the Model-Specific Code:
After running the architecture block, go to the corresponding code block at the bottom of the notebook for that model and run it.

Notes
-----
1. No pretrained models or external datasets are used.
2. Results are reproducible if cells are run in the intended order.
3. The project performs 3-class classification using both time series and tabular data.
