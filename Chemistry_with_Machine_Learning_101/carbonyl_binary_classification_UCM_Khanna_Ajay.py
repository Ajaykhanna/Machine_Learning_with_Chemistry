# **Speaker**: Ajay Khanna
# **Place**: JCTC Meeting, CBC, UC Merced
# Modified Version Date: Feb.26.2023
# Adapted for UC Merced Audience
# Objectives
# - Gain proficiency in reading and modifying Python code in the Google Colab/Jupyter notebook environment
# - Build machine learning binary classification models that predict the presence of a carbonyl group using IR spectroscopy data
# - Learn how to load and preprocess data
# - Learn how to read IR spectra
# - Learn array Slicing and indexing with Pandas
# - Learn to deal with bias in data and how to use synthetic data (cautiously)
# - Learn the use of normalization and cut-off
# - Learn how to train a DecisionTree ML model with your data
# - Learn how to use predict values on a trained DecisionTree ML model
# - Learn how to measure the accuracy of your model
# - Learn how to use RDKit

# This Jupyter Notebook was inspired by the following article. I used this
# article to understand the basics of ML and its application in Chemistry.
# I also taught my colleagues about this article and introduced them to
# Google Colab and how to run this notebook on their laptops:

# # ------- Journal of Chemical Education Article --------
# If you would like to learn more about the decision tree and other simple machine-learning models,
# below is the link to the paper that helped me get started. Some suggestions on how to get the most
# out of this paper:
#
# 1. To quickly start, read the .doc file from the SI, if you are comfortable with decision trees now
# 2. If you want to build chemical intuition, go through the main paper and then read the .doc file
# 3. The Jupyter Notebook they shared is slightly outdated, and most likely, it will give you errors
# if you try to run it. There are two solutions to this: [Ask me or go to the GitHub repository of the authors,
# https://github.com/elizabeththrall/MLforPChem/tree/main/MLforvibspectroscopy]
#
# Reference:
# Machine Learning for Functional Group Identification in Vibrational Spectroscopy:
# A Pedagogical Lab for Undergraduate Chemistry Students, https://pubs.acs.org/doi/full/10.1021/acs.jchemed.1c00693

# Loading Python Libraries
import pandas as pd
import numpy as np

# for normalization
from sklearn import preprocessing

# for visualization
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# For Machine Learning Model: Decision Tree
from sklearn.tree import DecisionTreeClassifier

# for data imbalance, SMOTE
from imblearn.over_sampling import SMOTE
from scipy import stats # Note: stats is imported but not explicitly used in the provided selection.

# to calculate the performance of the models
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

# ## Installing RDKit Module
# - To look at the molecule structure, we will use the `RDKit` [module](https://www.rdkit.org/)
# - The two code blocks below will install RDKit in Google Colab
#

import sys

# The following line is for installing RDKit in Google Colab or Jupyter environments.
# If running locally, you might need to install RDKit via other means (e.g., conda).
#!time pip install rdkit-pypi

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import IPythonConsole # Used for displaying molecules in Jupyter
except ImportError:
    print(
        "RDKit not found. Please ensure it is installed. "
        "If you are in Google Colab, the cell above this one attempts to install it. "
        "You might need to restart the runtime after installation and run this cell again."
    )
    # exit() # Exiting here can be disruptive in a notebook. Better to let the user handle it.

# # Get Data
# Now let's load in the training and test datasets.
# Download data from here:
# 1. Training data: https://drive.google.com/file/d/1JXBRjnjI5hMmXeSVCajEsFz0mlqCqOND/view?usp=share_link
# 2. Test data: https://drive.google.com/file/d/12tER4TtH3Eeojn2mccZXwdlYluKqRCxK/view?usp=share_link
#
#

# load the training data and save it in the variable "train"
# index_col=0 means the first column in the CSV is used as the row index
train = pd.read_csv("data/binary_train.csv", index_col=0)
# load the test data and save it in the variable "test"
test = pd.read_csv("data/binary_test.csv", index_col=0)

# Let's see what these data look like. You can display the current contents of a
# variable by entering its name and executing the cell:
#

# display the contents of the variable "train"
train

# * Each row contains data for a different molecule
# * The numbers to the left the first column (**0, 1, ...**) represent the index of each row
# * The first column ("SMILES") contains the molecule SMILES string (a way to write chemical structures as text)
# * The second column ("name") contains the molecule name
# * The third column ("label") contains a number indicating whether the molecule does (**1**) or does not (**0**) contain a carbonyl group
# * The numbers at the top of the remaining columns (**500, 502, ..., 3998, 4000**) represent the vibrational frequency in wavenumbers, and the numbers below each frequency represent the vibrational intensity of each molecule at that frequency
#
# We say that the vibrational intensity at each frequency is an **attribute** or **feature**. These terms refer to a property that can take on different values for different members of the dataset.

# ## Data Selection with Pandas
# We often need to look at our data in a particular way to store or extract information out it.
# To achieve that we use what is know as data selection or slicing. The task is performed as follow:
# - `iloc[row index, column index] `is used for position based data selection (integer-location based)
# - `:` is used for selecting a range of index values
# - Note that in Python, index values start from `0` instead of `1`
#
# For example:
# - `iloc[1:3,0]` : select row indices 1 to 2 (i.e., second and third rows) and the first column
# - `iloc[:,0]` : select all rows and the first column
# - `iloc[:,2:5]`: select all rows and column indices 2 to 4 (i.e., third through fifth columns)

# Extract first 3 rows with 10 columns from the training data
train.iloc[0:3, 0:10]

# # Plotting Spectra
# Before continuing, let's look at the spectra of a few molecules to see what they look like.
#
# Note that the index values below refer to the row numbers in the training data DataFrame. For example, `idx_notCarbonyl=1`
# selects the molecule in row 1 of the training data. If you want to select a different molecule,
# change the line of code to use its row index.

# change the index values below to pick molecules with and without a carbonyl
# Row index 0 in 'train' is a molecule w/o C=O, row index 1 is a molecule w/ C=O (as per original comments)
idx_notCarbonyl = 0
idx_hasCarbonyl = 1

# Get the data for the two molecules
# .set_index("name") makes the 'name' column the row index for easier identification in the plot legend.
# .iloc[idx_notCarbonyl, 3:] selects the specified row and all columns from the 4th one onwards (spectral data).
notCarbonyl = train.set_index("name").iloc[
    idx_notCarbonyl, 3:
]
hasCarbonyl = train.set_index("name").iloc[
    idx_hasCarbonyl, 3:
]
# plot the spectra
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=notCarbonyl.index, y=notCarbonyl, name=notCarbonyl.name, mode="markers"
    )
)

fig.add_trace(
    go.Scatter(
        x=hasCarbonyl.index, y=hasCarbonyl, name=hasCarbonyl.name, mode="markers"
    )
)

# Graph Layout
fig.update_layout(
    title="Intensities over frequency (Original Data)",
    title_x=0.5,
    xaxis_title="Frequencies (cm<sup>-1</sup>)", # Using _title for newer Plotly versions
    yaxis_title="Intensities"
)
fig.show()


# # Data Preprocessing | Standardization of the Data
# Before carrying out the machine learning analysis, we will need to preprocess the data to
# put it in a standard form. There are several steps involved: normalization, thresholding,
# splitting attribute and label, and data balancing.

# ## Normalization
# In practice, different IR spectra may be recorded at different molecular
# concentrations, so the absolute intensities may not be directly comparable
# (assuming no aggregation effects). Therefore we will **normalize** the data
# before carrying out the analysis.
#
# We will apply a type of normalization called **min-max normalization** to each "instance" (i.e., molecule) and update the data.
# - For each molecule, the spectral intensities will be scaled to **range from 0 to 1**
# - We will use the [MinMaxScaler()](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) method
#
# We will define a function called `df_normalize` to carry out this normalization:
# - The first argument is the DataFrame to be normalized.
# - The second argument (`intensity_start_col`) specifies where the spectral data columns begin.


# define a function to perform min-max normalization
def df_normalize(df, intensity_start_col=3):
  """
  Normalizes the intensity data in a DataFrame to a range of 0 to 1 for each molecule.

  This process scales the intensity values for each spectrum (row) so that the
  minimum intensity becomes 0 and the maximum becomes 1. This helps make
  spectra comparable even if measured at different concentrations.

  Args:
    df (pd.DataFrame): The DataFrame containing the spectral data.
                       Rows are molecules, and columns include names, labels, and intensities.
    intensity_start_col (int, optional): The column index where the
                                         intensity data begins. Defaults to 3,
                                         meaning columns from index 3 onwards are treated as intensities.

  Returns:
    None: The DataFrame `df` is modified in place.
  """
  min_max_scaler = preprocessing.MinMaxScaler()
  # MinMaxScaler usually works on columns. Our intensities for one spectrum are in a row.
  # So, we take the intensity part of the DataFrame:
  # 1. `.iloc[:, intensity_start_col:]` selects all rows and columns from `intensity_start_col` onwards.
  # 2. `.T` transposes this selection, so each spectrum (originally a row) becomes a column.
  # 3. `min_max_scaler.fit_transform()` then scales each column (each spectrum) to the 0-1 range.
  # 4. `.T` transposes the result back, so the scaled spectra are rows again.
  # This scaled data then replaces the original intensity data in the DataFrame.
  df.iloc[:, intensity_start_col:] = min_max_scaler.fit_transform(
      df.iloc[:, intensity_start_col:].T
  ).T


# use the function to normalize the training and test data
df_normalize(train)
df_normalize(test)

# ## Apply Threshold
# We expect that intensities near 0 won't provide much useful information for the classification.
# Therefore we will choose a threshold intensity and set all intensity values below the threshold
# equal to 0. This can help reduce noise.
#
# Let's look at the spectra of a few molecules (now normalized) and then choose the threshold.
# (Again you can choose which spectra to plot by changing the index values.)

# change the index values below to pick molecules with and without a carbonyl
idx_notCarbonyl = 0
idx_hasCarbonyl = 1
# get the data for the two molecules (from the now-normalized 'train' DataFrame)
notCarbonyl = train.set_index("name").iloc[idx_notCarbonyl, 3:]
hasCarbonyl = train.set_index("name").iloc[idx_hasCarbonyl, 3:]
# plot the spectra
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=notCarbonyl.index, y=notCarbonyl, name=notCarbonyl.name, mode="markers"
    )
)
fig.add_trace(
    go.Scatter(
        x=hasCarbonyl.index, y=hasCarbonyl, name=hasCarbonyl.name, mode="markers"
    )
)
# Graph Layout
fig.update_layout(
    title="Intensities over frequency (After Normalization)",
    title_x=0.5,
    xaxis_title="Frequencies (cm<sup>-1</sup>)",
    yaxis_title="Normalized Intensities"
)
fig.show()

# We will use a default value of `threshold=0.2` to start, but you can change this
# value later to see how it affects model performance:

# set threshold value
threshold = 0.2

# We will define a function called `applyThreshold` to apply the threshold chosen above.
# - This function uses the numpy "where" method: `np.where(condition, value_if_true, value_if_false)`
# - The first argument is the DataFrame.
# - The second argument (`intensity_start_col`) specifies where the spectral data columns begin.

def applyThreshold(dataframe, intensity_start_col=3):
    """
    Sets spectral intensity values below a defined 'threshold' to zero.

    This helps to remove noise or very small, potentially irrelevant signals
    from the spectra. The `threshold` variable must be defined globally
    (outside this function) before calling this function.

    Args:
      dataframe (pd.DataFrame): The DataFrame containing the spectral data to be thresholded.
      intensity_start_col (int, optional): The column index where the
                                           intensity data begins. Defaults to 3.

    Returns:
      None: The DataFrame `dataframe` is modified in place.
    """
    # Select the intensity data part of the DataFrame
    intensity_data = dataframe.iloc[:, intensity_start_col:]
    
    # Apply the threshold:
    # If a value in intensity_data is less than the global 'threshold', replace it with 0.
    # Otherwise, keep the original value.
    dataframe.iloc[:, intensity_start_col:] = np.where(
        intensity_data < threshold,  # Condition
        0,                           # Value if condition is true
        intensity_data               # Value if condition is false
    )

# use the "applyThreshold" function to apply the threshold to the training and test data
applyThreshold(train)
applyThreshold(test)

# Let's see how the intensities changed after applying the threshold:

# change the index values below to pick molecules with and without a carbonyl
idx_notCarbonyl = 0
idx_hasCarbonyl = 1
# get the data for the two molecules (from the thresholded 'train' DataFrame)
hasCarbonyl = train.set_index("name").iloc[idx_hasCarbonyl, 3:]
notCarbonyl = train.set_index("name").iloc[idx_notCarbonyl, 3:]
# plot the spectra
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=notCarbonyl.index, y=notCarbonyl, name=notCarbonyl.name, mode="markers"
    )
)
fig.add_trace(
    go.Scatter(
        x=hasCarbonyl.index, y=hasCarbonyl, name=hasCarbonyl.name, mode="markers"
    )
)
# Graph Layout
fig.update_layout(
    title="Intensities over frequency (After Thresholding)",
    title_x=0.5,
    xaxis_title="Frequencies (cm<sup>-1</sup>)",
    yaxis_title="Normalized Intensities (Thresholded)"
)
fig.show()

# ## Split Attribute and Label
# Notice that the training and test DataFrames contain the molecule name and label in addition to the spectral data.
# For machine learning, we need to separate:
# - **Features (X)**: The input data our model will learn from (here, the spectral intensities).
# - **Labels (Y)**: The output we want to predict (here, whether a carbonyl group is present: 1 for yes, 0 for no).
#
# Define a function to split the labels and features:
# - The first argument is the DataFrame.
# - `feature_start_col` and `feature_end_col` define the columns for X.
# - `label_start_col` and `label_end_col` define the column(s) for Y.

def splitXY(dataframe, feature_start_col=3, feature_end_col=None, label_col_index=2):
  """
  Separates a DataFrame into features (X) and labels (Y) for machine learning.

  Features (X) are the input data for the model (e.g., spectral intensities).
  Labels (Y) are what the model tries to predict (e.g., presence of a carbonyl group).

  Args:
    dataframe (pd.DataFrame): The input DataFrame (e.g., 'train' or 'test' data).
    feature_start_col (int, optional): Starting column index for features (X).
                                       Defaults to 3 (assuming SMILES, name, label are first 3 cols).
    feature_end_col (int or None, optional): Ending column index (exclusive) for features (X).
                                             Defaults to None (selects all columns from feature_start_col to the end).
    label_col_index (int, optional): Column index for the labels (Y).
                                     Defaults to 2 (assuming the label is in the 3rd column).

  Returns:
    tuple: A tuple containing two NumPy arrays:
      - X (np.ndarray): The feature matrix (e.g., spectral intensities).
                        Shape: (number_of_molecules, number_of_frequencies).
      - Y (np.ndarray): The label vector (e.g., 0 or 1 for carbonyl presence).
                        Shape: (number_of_molecules,).
  """
    # Extract feature columns (e.g., spectral intensities)
    # .iloc[:, feature_start_col:feature_end_col] selects all rows and specified columns for features.
    X_df = dataframe.iloc[:, feature_start_col:feature_end_col]
    # Convert the features DataFrame to a NumPy array. Machine learning models in scikit-learn
    # typically expect NumPy arrays as input.
    X = X_df.values

    # Extract the label column
    # .iloc[:, label_col_index] selects all rows and the single specified column for labels.
    Y_series = dataframe.iloc[:, label_col_index]
    # Convert the labels Series to a 1D NumPy array (a vector).
    # .values gets the underlying NumPy data.
    # .ravel() is used to ensure Y is a 1D array, e.g., [0, 1, 0, ...], not [[0], [1], [0], ...].
    Y = Y_series.values.ravel()
    
    # Ensure labels are integers (e.g., 0 or 1), as they might be read as floats initially.
    Y = Y.astype(float).astype(int)
    return X, Y

# Apply the function to the preprocessed train and test data.
# For 'train' data: X will contain spectral intensities, Y will contain carbonyl labels.
# For 'test' data: X_test will contain spectral intensities, Y_test will contain carbonyl labels.
X, Y = splitXY(train, feature_start_col=3, label_col_index=2)
X_test, Y_test = splitXY(test, feature_start_col=3, label_col_index=2)


# ## Data Balancing
# Let's visualize the data distribution with a pie chart to see if data
# are imbalanced. Imbalanced means that there are unequal numbers
# of molecules in the two classes (with and without a carbonyl).

# get the total number of molecules in the training data's labels
total = len(Y)
# determine how many contain a carbonyl (label is 1)
# Since labels are 0 or 1, sum(Y) gives the count of 1s.
label1_count = Y.sum()
# find the number without a carbonyl (label is 0) by subtraction
label0_count = total - label1_count

# plot the data
data_counts = [label1_count, label0_count]
my_labels = "Has Carbonyl (1)", "No Carbonyl (0)"
fig = go.Figure(data=[go.Pie(labels=my_labels, values=data_counts, hole=.3)])
fig.update_layout(title_text="Original Training Data Distribution (Before SMOTE)", title_x=0.5)
fig.show()

# Molecules without a carbonyl dominate the training set, so the classes are imbalanced.
# ### SMOTE (Synthetic Minority Oversampling Technique)
# Imbalanced training data can sometimes lead to poor classification performance because
# the model might simply learn to predict the majority class most of the time. To address
# this, we can use a technique called [SMOTE]
# (https://en.wikipedia.org/wiki/Oversampling_and_undersampling_in_data_analysis#SMOTE).
# SMOTE generates new *synthetic* instances of the minority class (here, molecules with carbonyls)
# by interpolating between existing instances in the feature space.
# (Note: For some datasets where classes are very distinct, balancing might not always be necessary,
# but it's a common and often useful step.)

# Define SMOTE method
# sampling_strategy='minority' tells SMOTE to resample only the minority class.
sm = SMOTE(sampling_strategy="minority", random_state=42) # random_state for reproducibility
# Apply SMOTE to the training data (X features, Y labels)
# fit_resample will generate new samples for the minority class in X and adjust Y accordingly.
X_resampled, Y_resampled = sm.fit_resample(X, Y)

# Let's check the distribution after SMOTE
total_resampled = len(Y_resampled)
label1_resampled_count = Y_resampled.sum()
label0_resampled_count = total_resampled - label1_resampled_count

data_resampled_counts = [label1_resampled_count, label0_resampled_count]
fig = go.Figure(data=[go.Pie(labels=my_labels, values=data_resampled_counts, hole=.3)])
fig.update_layout(title_text="Training Data Distribution (After SMOTE)", title_x=0.5)
fig.show()

# Now the training data (X_resampled, Y_resampled) should be balanced.
# We can plot one of the new synthetic carbonyl-containing spectra
# for comparison to a real carbonyl-containing spectrum.
# The synthetic spectra are added at the end of the X_resampled array.

# Index for a real carbonyl spectrum from the original dataset (e.g., the one at original index 1)
idx_realCarbonyl = 1 # This was 'hasCarbonyl' in earlier plots
# Index for a synthetic carbonyl spectrum.
# Original X had 'len(X)' samples. After SMOTE, X_resampled is larger.
# len(X) would be the index of the first synthetic sample if all original samples are kept at the beginning.
idx_synCarbonyl = len(X) # This assumes SMOTE appends new samples.

# Get the data for the two molecules
spectrum_realCarbonyl = X_resampled[idx_realCarbonyl, :] # From the resampled data
spectrum_synCarbonyl = X_resampled[idx_synCarbonyl, :]   # A synthetic sample

# Get the frequencies for plotting (these are the column headers from the original spectral data)
# Assuming frequencies start from column 3 in the 'train' DataFrame
frequencies = train.columns[3:].astype(float) # Ensure frequencies are numeric for plotting

# Generate the plot
fig = go.Figure()
fig.add_trace(
    go.Scatter(x=frequencies, y=spectrum_realCarbonyl, name=f"Real C=O (Original Index {idx_realCarbonyl})", mode="lines+markers")
)
fig.add_trace(
    go.Scatter(x=frequencies, y=spectrum_synCarbonyl, name=f"Synthetic C=O (Index {idx_synCarbonyl} in Resampled Data)", mode="lines+markers")
)

# Graph Layout
fig.update_layout(
    title="Comparison of Real vs. Synthetic Carbonyl Spectra",
    title_x=0.5,
    xaxis_title="Frequencies (cm<sup>-1</sup>)",
    yaxis_title="Normalized Intensities (Thresholded)"
)
fig.show()

# # Building Machine Learning Models

# **Decision Tree**
#
# A Decision Tree is a type of machine learning model that makes predictions
# by learning simple decision rules inferred from the data features. It looks like
# a tree structure, where each internal node represents a "test" on an attribute
# (e.g., is intensity at 1700 cm-1 > 0.5?), each branch represents the outcome
# of the test, and each leaf node represents a class label (e.g., carbonyl or no carbonyl).
# - [scikit-learn DecisionTreeClassifier documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
#

# ## Training a Decision Tree Model
# Let's train a Decision Tree model using our balanced and preprocessed training data (X_resampled, Y_resampled).
# "Training" or "fitting" the model means it learns the patterns from this data.

# Initialize the Decision Tree classifier. We can use default parameters for now.
# random_state ensures that if you run the code again, the tree will be built the same way (if data is the same).
dt_clf = DecisionTreeClassifier(random_state=42)

# Fit the model using the resampled (balanced) training dataset
dt_clf.fit(X_resampled, Y_resampled)
print("Decision Tree model trained successfully.")

# # Testing Machine Learning Models
# Now that we have trained our machine learning model, we need to evaluate its performance
# on data it hasn't seen before. This is what the "test set" (X_test, Y_test) is for.
#
# 1. **Label Prediction**:
#    We use the trained model (`dt_clf`) to predict labels for the test set (`X_test`).
#    These predictions will be stored in `Y_pred`.
#
# 2. **Model Evaluation**:
#    We compare the model's predictions (`Y_pred`) with the true labels (`Y_test`).
#    Common metrics for binary classification (like ours) include:
#    - **Accuracy**: The proportion of total predictions that were correct.
#      (TP + TN) / (TP + TN + FP + FN)
#    - **Sensitivity (Recall)**: The proportion of actual positive cases (molecules *with* a carbonyl)
#      that were correctly identified by the model. TP / (TP + FN)
#    - **Specificity**: The proportion of actual negative cases (molecules *without* a carbonyl)
#      that were correctly identified. TN / (TN + FP)
#    (TP=True Positives, TN=True Negatives, FP=False Positives, FN=False Negatives)

# ## Test the Decision Tree Model
# Let's use the trained Decision Tree model for label prediction on the test set.
# Then we'll calculate and display its accuracy, sensitivity, and specificity.

# Use the trained Decision Tree model to predict labels for the test dataset (X_test)
Y_pred = dt_clf.predict(X_test)

# Calculate performance metrics by comparing predicted labels (Y_pred) to actual labels (Y_test)
dt_accuracy = accuracy_score(Y_test, Y_pred)
dt_sensitivity = recall_score(Y_test, Y_pred, pos_label=1) # pos_label=1 for "carbonyl present"
dt_specificity = recall_score(Y_test, Y_pred, pos_label=0) # pos_label=0 for "carbonyl absent"

# Display the metrics, rounded to 2 decimal places for readability
print(f"Decision Tree Model Performance on Test Data:")
print(f"  Accuracy:    {dt_accuracy:.2f}")
print(f"  Sensitivity (Recall for C=O): {dt_sensitivity:.2f}")
print(f"  Specificity (Recall for no C=O): {dt_specificity:.2f}")


# # FP / FN Group Analysis
# Now let's look more carefully at the False Positives (FPs) and False Negatives (FNs).
# - **False Positive (FP)**: Model predicted "carbonyl present" (1), but it was actually "absent" (0).
# - **False Negative (FN)**: Model predicted "carbonyl absent" (0), but it was actually "present" (1).
# Analyzing these errors can give insights into why the model makes mistakes.

# (The Y_pred variable already holds the predictions from the Decision Tree model from the cell above)

# Create empty lists to store the original indices (row numbers in the 'test' DataFrame) of FPs and FNs
fp_indices = []
fn_indices = []

# Go through all predictions for the test set
for i in range(len(Y_test)):
    # Check if the prediction is wrong
    if Y_pred[i] != Y_test[i]:
        # If wrong and true label was 0 (no carbonyl), it's a False Positive
        if Y_test[i] == 0: # Model predicted 1, but truth was 0
            fp_indices.append(i)
        # If wrong and true label was 1 (has carbonyl), it's a False Negative
        elif Y_test[i] == 1: # Model predicted 0, but truth was 1
            fn_indices.append(i)

print(f"Found {len(fp_indices)} False Positives and {len(fn_indices)} False Negatives.")

# ## Utilizing `RDKit` to get the molecule structure
#
# We can use the `RDKit` library to display the chemical structure of a molecule.
# To do this, we need the SMILES string of the molecule. SMILES is a text-based
# way to represent a chemical structure.
# Let's get the SMILES strings and names for our FP and FN molecules from the 'test' DataFrame.

# Get SMILES strings and names for False Positives
# .iloc[fp_indices, 0] gets the SMILES (column 0) for rows at fp_indices
fp_smiles = test.iloc[fp_indices, 0].values
fp_names = test.iloc[fp_indices, 1].values # Column 1 is 'name'

# Get SMILES strings and names for False Negatives
fn_smiles = test.iloc[fn_indices, 0].values
fn_names = test.iloc[fn_indices, 1].values

# Convert SMILES strings to RDKit molecule objects
# Chem.MolFromSmiles() parses a SMILES string and creates a molecule object
fp_mol_objects = [Chem.MolFromSmiles(s) for s in fp_smiles]
fn_mol_objects = [Chem.MolFromSmiles(s) for s in fn_smiles]

# Set molecule names as a property for display in RDKit plots
for i, mol in enumerate(fp_mol_objects):
    if mol: # Check if mol object was created successfully
        mol.SetProp("_Name", fp_names[i])

for i, mol in enumerate(fn_mol_objects):
    if mol:
        mol.SetProp("_Name", fn_names[i])

# Now let's display tables showing the index, name, and SMILES string of any FPs
# and FNs. Note that if the model performance was very good, there might not be
# any FPs and/or FNs, so the table(s) will be empty in that case.

# Display a table showing the index, SMILES string, and name of all FPs
print("\n" + "\033[1m" + "False Positives (FPs): Predicted Carbonyl, Actual No Carbonyl" + "\033[0m")
if fp_indices:
    display(test.iloc[fp_indices, 0:2]) # Display SMILES and name columns
else:
    print("No False Positives found.")

# Display a table showing the index, SMILES string, and name of all FNs
print("\n" + "\033[1m" + "False Negatives (FNs): Predicted No Carbonyl, Actual Carbonyl" + "\033[0m")
if fn_indices:
    display(test.iloc[fn_indices, 0:2])
else:
    print("No False Negatives found.")


# **FP Group Structures**
# Display the molecular structures of all False Positives
if fp_mol_objects:
    img1 = Draw.MolsToGridImage(
        [m for m in fp_mol_objects if m], # Filter out None if SMILES parsing failed
        molsPerRow=4,
        subImgSize=(200, 200),
        legends=[mol.GetProp("_Name") for mol in fp_mol_objects if mol]
    )
    display(img1)
else:
    print("No False Positive structures to display.")


# **FN Group Structures**
# Display the molecular structures of all False Negatives
if fn_mol_objects:
    img2 = Draw.MolsToGridImage(
        [m for m in fn_mol_objects if m], # Filter out None
        molsPerRow=4,
        subImgSize=(200, 200),
        legends=[mol.GetProp("_Name") for mol in fn_mol_objects if mol]
    )
    display(img2)
else:
    print("No False Negative structures to display.")


# ## Displaying FP or FN Spectrum
# Edit the molecule index below to display the spectrum of a particular FP
# or FN for inspection. The index should be one of the original indices from the 'test' DataFrame
# that you identified as an FP or FN.
# Do you see any spectral features that might explain the error?

# Example: To display the spectrum of a specific False Positive
# Replace `your_fp_index_here` with an actual index from the FP list above.
# For example, if `test.iloc[78]` was an FP: fp_idx_to_plot = 78
if fp_indices: # Check if there are any FPs
    fp_idx_to_plot = fp_indices[0] # Plot the first FP as an example
    print(f"\nDisplaying spectrum for FP: Index {fp_idx_to_plot}, Name: {test.iloc[fp_idx_to_plot, 1]}")
    
    # .iloc[fp_idx_to_plot, 3:] gets spectral data (from 4th col onwards) for the chosen molecule
    # .set_index("name") is not strictly needed here if we just plot, but good for consistency if name is used.
    fp_spectrum = test.iloc[fp_idx_to_plot, 3:]
    fp_name = test.iloc[fp_idx_to_plot, 1]

    fig_fp = go.Figure()
    fig_fp.add_trace(go.Scatter(x=fp_spectrum.index.astype(float), y=fp_spectrum.values, name=fp_name, mode="lines+markers"))
    fig_fp.update_layout(
        title=f"IR Spectrum of False Positive: {fp_name} (Index {fp_idx_to_plot})",
        title_x=0.5,
        xaxis_title="Frequencies (cm<sup>-1</sup>)",
        yaxis_title="Normalized Intensity (Thresholded)",
        showlegend=True
    )
    fig_fp.show()
else:
    print("\nNo False Positives to plot spectrum for.")

# Example: To display the spectrum of a specific False Negative
# Replace `your_fn_index_here` with an actual index from the FN list above.
if fn_indices: # Check if there are any FNs
    fn_idx_to_plot = fn_indices[0] # Plot the first FN as an example
    print(f"\nDisplaying spectrum for FN: Index {fn_idx_to_plot}, Name: {test.iloc[fn_idx_to_plot, 1]}")

    fn_spectrum = test.iloc[fn_idx_to_plot, 3:]
    fn_name = test.iloc[fn_idx_to_plot, 1]

    fig_fn = go.Figure()
    fig_fn.add_trace(go.Scatter(x=fn_spectrum.index.astype(float), y=fn_spectrum.values, name=fn_name, mode="lines+markers"))
    fig_fn.update_layout(
        title=f"IR Spectrum of False Negative: {fn_name} (Index {fn_idx_to_plot})",
        title_x=0.5,
        xaxis_title="Frequencies (cm<sup>-1</sup>)",
        yaxis_title="Normalized Intensity (Thresholded)",
        showlegend=True
    )
    fig_fn.show()
else:
    print("\nNo False Negatives to plot spectrum for.")


# # Conclusion
# We learned how to:
# - Use Google Colab/VS Code Jupyter notebook environment
# - Load and preprocess data (normalization, thresholding)
# - Read and visualize IR spectra
# - Slice and index data with Pandas
# - Understand and address data imbalance using SMOTE (Synthetic Minority Oversampling Technique)
# - Separate data into features (X) and labels (Y) for machine learning
# - Train a Decision Tree machine learning model
# - Use the trained model to make predictions on new data
# - Measure the model's performance using accuracy, sensitivity, and specificity
# - Analyze model errors (False Positives and False Negatives)
# - Use RDKit to visualize molecular structures
