================================
PIRVision Fog Presence Detection
================================
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