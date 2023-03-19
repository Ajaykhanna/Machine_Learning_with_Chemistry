# %% [markdown]
# ### **Speaker**: Ajay Khanna
# 
# ### **Place**: JCTC Meeting, CBC, UC Merced
# 
# Modified Version Date: Feb.26.2023
# 
# Adapted for UC Merced Audience

# %% [markdown]
# # Goals/Objectives
# - Gain proficiency in reading and modifying Python code in the Google Colab/Jupyter notebook environment
# - Build machine learning binary classification models that predict the presence of a carbonyl group using IR spectroscopy data
# - Learn how load and preprocess data
# - Lear how to read IR spectra
# - Learn array Slicing and indexing with Pandas
# - Learn to deal with bais in data and how to use sythetica data (cautiously)
# - Learn use of normalization and cut-off
# - Learn how to use train DecisionTree ML model with your data
# - Learn how to use predicit values on trained DecisionTree ML model
# - Learn how measure accuracy of your model
# - Learn how to use RdK!t 

# %% [markdown]
# ## Loading Python Libraries

# %%
%%capture
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
from scipy import stats

# to calculate the performance of the models
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

# %% [markdown]
# ## Installing RDKit Module
# - To look at the molecule structure, we will use the `RDKit` [module](https://www.rdkit.org/)
# - The two code blocks below will install RDKit in Google Colab
# 

# %%
import sys
#!time pip install rdkit-pypi

# %%
try:
  from rdkit import Chem
  from rdkit.Chem import Draw
  from rdkit.Chem.Draw import IPythonConsole
except ImportError:
  print('Stopping RUNTIME. Colaboratory will restart automatically. Please run again.')
  exit()

# %% [markdown]
# # Get Data
# Now let's load in the training and test datasets.
# Download data from here:
# 1. Training data: https://drive.google.com/file/d/1JXBRjnjI5hMmXeSVCajEsFz0mlqCqOND/view?usp=share_link
# 2. Test data: https://drive.google.com/file/d/12tER4TtH3Eeojn2mccZXwdlYluKqRCxK/view?usp=share_link
# 
# 

# %%
# load the training data and save it in the variable "train"
train=pd.read_csv('data/binary_train.csv',index_col=0)
# load the test data and save it in the variable "test"
test=pd.read_csv('data/binary_test.csv',index_col=0)

# %% [markdown]
# Let's see what these data look like. You can display the current contents of a variable by entering its name and executing the cell:
# 

# %%
# display the contents of the variable "train"
train

# %% [markdown]
# * Each row contains data for a different molecule
# * The numbers to the left the first column (**0, 1, ...**) represent the index of each row
# * The first column ("SMILES") contains the molecule SMILES string (more on that later)
# * The second column ("name") contains the molecule name
# * The third column ("label") contains a number indicating whether the molecule does (**1**) or does not (**0**) contain a carbonyl group
# * The numbers at the top of the remaining columns (**500, 502, ..., 3998, 4000**) represent the vibrational frequency in wavenumbers, and the numbers below each frequency represent the vibrational intensity of each molecule at that frequency
# 
# We say that the vibrational intensity at each frequency is an **attribute** or **feature**. These terms refer to a property that can take on different values for different members of the dataset.

# %% [markdown]
# ## Data Selection with Pandas
# We often need to look at our data in a particular way to store or extract information out it. To achieve that we use what is know as data selection or slicing. The task is performed as follow:
# - `iloc[row index, column index] `is used for position based data selection
# - `:` is used for selecting a range of index values
# - Note that in Python, index values start from `0` instead of `1`
# 
# For example:
# - `iloc[1:3,0]` : select row indices 1 to 2 (i.e., second and third rows) and the first column
# - `iloc[:,0]` : select all rows and the first column
# - `iloc[:,2:5]`: select all rows and column indices 2 to 4 (i.e., third through fifth columns)

# %%
# Extract first 3 rows with 10 columns from the training data
train.iloc[0:3,0:10]

# %% [markdown]
# # Plotting Spectra
# Before continuing, let's look at the spectra of a few molecules to see what they look like.
# 
# Note that the index values below refer to the row numbers in the training data DataFrame. For example, `idx_notCarbonyl=1` selects the molecule in row 1 of the training data, which is hexanal. If you want to select 12-selenazole in row 3 instead, change the line of code to read `idx_notCarbonyl=3`.

# %%
# change the index values below to pick molecules with and without a carbonyl
# 0: w/o C=O and 1: w/t C=O
idx_notCarbonyl = 0
idx_hasCarbonyl = 1
# get the data for the two molecules
notCarbonyl = train.set_index('name').iloc[idx_notCarbonyl,3:] # C!=O Object dataset: X: Frequencies, Y: Intensities, Name: Mol name
hasCarbonyl = train.set_index('name').iloc[idx_hasCarbonyl,3:] # C=O Object dataset: X: Frequencies, Y: Intensities, Name: Mol name
# plot the spectra
fig = go.Figure()
fig.add_trace(go.Scatter(x=notCarbonyl.index,
                         y=notCarbonyl,
                         name=notCarbonyl.name,
                         mode='markers'))

fig.add_trace(go.Scatter(x=hasCarbonyl.index,
                         y=hasCarbonyl,
                         name=hasCarbonyl.name,
                         mode='markers'))

# Graph Layout
fig.update_layout(title='Intensities over frequency', title_x=0.5,
                  xaxis = dict(title='$Frequencies (cm^{-1})$'),
                  yaxis = dict(title='Normalized Intensities'))

# %% [markdown]
# # Data Preprocessing | Standardization of the Data
# Before carrying out the machine learning analysis, we will need to preprocess the data to put it in a standard form. There are several steps involved: normalization, thresholding, splitting attribute and label, and data balancing.

# %% [markdown]
# ## Normalization
# In practice, different IR spectra may be recorded at different molecular concentrations, so the absolute intensities may not be directly comparable (assuming no aggregation effects). Therefore we will **normalize** the data before carrying out the analysis.
# 
# We will apply a type of normalization called **min-max normalization** to each "instance" (i.e., molecule) and update the data.
# - For each molecule, the spectral intensities will be scaled to **range from 0 to 1**
# - We will use the [MinMaxScaler()](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) method
# 
# We will define a function called `df_normalize` to carry out this normalization:
# - The first argument in the parentheses following the function name represents the data to be normalized
# - The second argument represents the column index where the frequency data start. If you don't provide this argument, the function uses a default value of `3`.
# - As an example, if the frequency data in the variable `ex_data` starts in column `4`, you would write: `df_normalize(ex_data,4)`

# %%
# define a function to perform min-max normalization
def df_normalize(df,i=3):
  """
  i = 3:, colmun index for intensities (three and onwards in this case)
  apply min-max_scaler to each rows
  since min-max scaler originally applies to columns, 
  we will use transposed data and then update the data with transposed result
  """
  min_max_scaler = preprocessing.MinMaxScaler()
  df.iloc[:,i:] = min_max_scaler.fit_transform(df.iloc[:,i:].T).T # Row(Start) --> Cols(Apply Transform) --> Row (End)

# %%
# use the functional to normalize the training and test data
df_normalize(train)
df_normalize(test)

# %% [markdown]
# ## Apply Threshold
# We expect that intensities near 0 won't provide much useful information for the classification. Therefore we will choose a threshold intensity and set all intensity values below the threshold equal to 0.
# 
# Let's look at the spectra of a few molecules and then choose the threshold. (Again you can choose which spectra to plot by changing the index values.)

# %%
# change the index values below to pick molecules with and without a carbonyl
idx_notCarbonyl=0
idx_hasCarbonyl=1
# get the data for the two molecules
notCarbonyl=train.set_index('name').iloc[idx_notCarbonyl,3:] 
hasCarbonyl=train.set_index('name').iloc[idx_hasCarbonyl,3:]
# plot the spectra
fig = go.Figure()
fig.add_trace(go.Scatter(x=notCarbonyl.index,
                         y=notCarbonyl,
                         name=notCarbonyl.name,
                         mode='markers'))

fig.add_trace(go.Scatter(x=hasCarbonyl.index,
                         y=hasCarbonyl,
                         name=hasCarbonyl.name,
                         mode='markers'))
# Graph Layout
fig.update_layout(title='Intensities over frequency', title_x=0.5,
                  xaxis = dict(title='$Frequencies (cm^{-1})$'),
                  yaxis = dict(title='Normalized Intensities'))

# %% [markdown]
# We will use a default value of `threshold=0.2` to start, but you can change this value later to see how it affects model performance:

# %%
# set threshold value
threshold=0.2

# %% [markdown]
# We will define a function called `applyThreshold` to apply the threshold chosen above to the training and test data:
# - This function uses the **numpy "where"** method to replace intensity values below the threshold with the value 0
# - The first argument in the parentheses following the function name represents the data to be thresholded
# - The second argument `i` represents the column index where the frequency data start. If you don't provide this argument, the function uses a default value of `3`.
# - As an example, if the frequency data in the variable `ex_data` starts in column `4`, you would write: `applyThreshold(ex_data,4)`

# %%
# Defining a function that will apply the threshold to the dataframe. The threshold is defined above.
def applyThreshold(dataframe,i=3):
  """
  i is the position of the start of the attributes
  """
  dataframe.iloc[:,i:]=np.where((dataframe.iloc[:,i:]< threshold),0,dataframe.iloc[:,i:])

# use the "applyThreshold" function to apply the threshold to the training and test data
applyThreshold(train)
applyThreshold(test)

# %% [markdown]
# Let's see how the intensities changed after applying the threshold:

# %%
# change the index values below to pick molecules with and without a carbonyl
idx_notCarbonyl=0
idx_hasCarbonyl=1
# get the data for the two molecules
hasCarbonyl=train.set_index('name').iloc[idx_hasCarbonyl,3:] # picked
notCarbonyl=train.set_index('name').iloc[idx_notCarbonyl,3:] # picked 
# plot the spectra
fig = go.Figure()
fig.add_trace(go.Scatter(x=notCarbonyl.index,
                         y=notCarbonyl,
                         name=notCarbonyl.name,
                         mode='markers'))

fig.add_trace(go.Scatter(x=hasCarbonyl.index,
                         y=hasCarbonyl,
                         name=hasCarbonyl.name,
                         mode='markers'))

# Graph Layout
fig.update_layout(title='Intensities over frequency', title_x=0.5,
                  xaxis = dict(title='$Frequencies (cm^{-1})$'),
                  yaxis = dict(title='Normalized Intensities'))

# %% [markdown]
# ## Split Attribute and Label
# Notice that the training and test DataFrames contain the molecule name and label in addition to the spectral data. Now we need to separate the information about whether or not the molecule contains a carbonyl from the spectral intensities. We will do this by creating two new variables, X and Y:
# 
# X is an **attribute** martrix, all the **normalized intensities**
# 
# Y is a **label** vector which is the **presence of a carbonyl group**. **If the molecule has a carbonyl then Y = 1; if it doesn't, then Y = 0**.
# 
# Define a function to split the labels and attributes:
# - The first argument in the parentheses following the function name represents the data to be split
# - The second argument `start_X` represents the column index where the normalized intensity data start. If you don't provide this argument, the function uses a default value of `3`.
# - The third argument `end_X` represents the column index where the normalized intensity data end. If you don't provide this argument, the function uses a default value of `None`.
# - The fourth argument `start_Y` represents the column index where the label data start. If you don't provide this argument, the function uses a default value of `2`.
# - The fifth argument `end_Y` represents the column index where the frequency data end. If you don't provide this argument, the function uses a default value of `3`.
# - As an example, if the frequency data in the variable `ex_data` start in column `5` and the label data are in columns `3` and `4`, you would write: `splitXY(ex_data,5,None,3,5)`

# %%
# define a function to split the column containing the label from the columns containing the attributes
def splitXY(dataframe,start_X=3,end_X=None,start_Y=2,end_Y=3):
  X=dataframe.iloc[:,start_X:end_X] # Pick Normalized intensities of all molecules
  # since current X is a dataframe structure, we use ".value" to only get values
  X=X.values # Convert dataframe to Matrix or list of list
  Y=dataframe.iloc[:,start_Y:end_Y]
  # since current Y is a dataframe structure, we use ".value" to only get values
  Y=Y.values.ravel() # If .value(), it will be a list of list, ravel makes its a vector
  # this makes sure all the labels are integers
  Y=Y.astype(float).astype(int)
  return X,Y

# %%
# now apply the function to the normalized and thresholded train and test data
X,Y=splitXY(train)
X_test,Y_test=splitXY(test)

# %% [markdown]
# ## Data Balancing

# %% [markdown]
# Let's visualize the data distribution with a pie chart to see if data are imbalanced. Here, imbalanced means that there are unequal numbers of molecules in the two classes (with and without a carbonyl).:

# %%
# get the total number of molecules in the training data
total=len(Y) # [0,1,...]
# determine how many contain a carbonyl
label1=Y.sum() # [0+1+..], sum is made out of only 1's
# find the number without a carbonyl by subtraction
label0=total-label1
# plot the data
data=[label1, label0]
my_labels = 'Carbonyl','notCarbonyl'
fig = go.Figure(data=[go.Pie(labels=my_labels,
                             values=data)])
fig.update_layout(title='Original Data Distribution')
fig.show()

# %% [markdown]
# Molecules without a carbonyl dominate the training set, so the classes are imbalanced.

# %% [markdown]
# ### SMOTE (Synthetic minority oversampling technique)

# %% [markdown]
# Imbalanced training data can sometimes lead to poor classification performance because the model might simply learn to ignore the less common ("minority") class. To address this possibility, we will use a technique called [SMOTE](https://en.wikipedia.org/wiki/Oversampling_and_undersampling_in_data_analysis#SMOTE) which generates new instances of the minority class by interpolating between the existing instances. (Note that if the two classes are sufficiently distinct, as is the case here, a data balancing step may not be required -- but we'll do it anyway.)

# %%
# define SMOTE method
# For more details on SMOTE: https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
sm = SMOTE(sampling_strategy='minority')
# apply SMOTE to the training data
X, Y  = sm.fit_resample(X,Y)

# %%
# Again determine the number of molecules with and without carbonyl groups and visualize
total=len(Y)
label1=Y.sum()
label0=total-label1
data=[label1,label0]
my_labels = 'Carbonyl','notCarbonyl'
fig = go.Figure(data=[go.Pie(labels=my_labels,
                             values=data)])
fig.update_layout(title='Original Data Distribution')
fig.show()

# %% [markdown]
# Now the training data are balanced between the two classes. We can plot one of the new synthetic carbonyl-containing spectra for comparison to a real carbonyl-containing spectrum. (The synthetic spectrum will vary each time you run SMOTE.) Note that the synthetic spectra are stored at the end of the variable X, so any index value greater than the original length of the variable train corresponds to a synthetic spectrum.

# %%
# index values of a real and synthetic carbonyl (you can change these values to see other spectra)
idx_realCarbonyl = 1 # this selects the molecule in row 1 of the training data (hexanal)
idx_synCarbonyl = len(train) # this selects the first synthetic carbonyl spectrum
# get the data for the two molecules
spectrum_realCarbonyl = X[idx_realCarbonyl,:]
spectrum_synCarbonyl = X[idx_synCarbonyl,:]
# get the frequencies for plotting
frequencies = np.arange(500,4002,2)
# generate the plot
fig = go.Figure()

fig.add_trace(go.Scatter(x=frequencies,
                         y=spectrum_realCarbonyl,
                         name="Real C=O",
                         mode='markers'))

fig.add_trace(go.Scatter(x=frequencies,
                         y=spectrum_synCarbonyl,
                         name="Syn C=O",
                         mode='markers'))

# Graph Layout
fig.update_layout(title='Real vs Synthetics Samples', title_x=0.5,
                  xaxis = dict(title='$Frequencies (cm^{-1})$'),
                  yaxis = dict(title='Normalized Intensities'))

# %% [markdown]
# # Building Machine Learning Models

# %% [markdown]
# **Decision Tree**
# 
# Decision Tree uses a tree-like model of decisions. It models the data in a tree structure, in which each leaf node corresponds to a class label and attributes are the tree’s internal nodes.
# - [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) documentation 
# 

# %% [markdown]
# ## Training a Decision Tree Model
# Let's start by training a Decision Tree model. Use the normalized and thresholded training data to fit the model.

# %%
# use the Decision Tree algorithm with the default parameters
dt_clf=DecisionTreeClassifier()
# fit the model using the training dataset
dt_clf.fit(X,Y)

# %% [markdown]
# # Testing Machine Learning Models
# Now that we have trained our machine learning models, we can use them on test data. There are a few steps in this process:
# 1. Label Prediction: 
# 
# We will use the fitted machine learning models to predict the labels (with or without carbonyl) for the test dataset. We will store the predictions made by the fitted models in the `Y_pred` list.
# 
# 2. Model Evaluation:
# 
# It's important to see how well the models performed. There are [a few ways](https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers) to assess the accuracy of a machine learning model:
# - Accuracy : the proportion of the total number of predictions that were correct
# - Sensitivity or Recall : the proportion of actual positive cases (here, molecules with a carbonyl) which are correctly identified
# - Specificity : the proportion of actual negative cases (here, molecules without a carbonyl) which are correctly identified
# 
# 3. False Negative and False Positive Group
# 
# Finally, we can analyze the error cases, where the model's prediction was wrong. A false positive (FP) is an outcome where the model incorrectly predicts the positive class. A false negative (FN) is an outcome where the model incorrectly predicts the negative class. Sometimes we might be able to understand why the prediction was incorrect for a particular molecule by:
# 
# - Looking at the molecular structure. (We will use the `RDkit` module for this task.)
# -Looking at the molecular spectrum.

# %% [markdown]
# ## Test the Decision Tree Model
# Now let's use the fitted Decision Tree model for label prediction. Then we'll analyze the performance by displaying the accuracy, sensitivity, and specificity.

# %%
# use the fitted Decision Tree model to predict the labels (with or without carbonyl) for the test dataset
Y_pred=dt_clf.predict(X_test)

# %%
# determine the accuracy, sensitivity, and specificity by comparing the predicted labels to the actual labels
dt_accuracy=np.round(accuracy_score(Y_test,Y_pred),2)
dt_sensitivity=np.round(recall_score(Y_test,Y_pred),2)
dt_specificity=np.round(recall_score(Y_test,Y_pred,pos_label=0),2)

# display the accuracy, sensitivity, and specificity
print("Accuracy: "+str(dt_accuracy)+" Sensitivity: "+str(dt_sensitivity)+" Specificity: "+str(dt_specificity))

# %% [markdown]
# # FP / FN Group Analysis
# Now let's look more carefully at the FPs and FNs. First we need to determine which predictions were incorrect and separate them into the two types of errors.
# 
# Note that for both the Decision Tree and the Random Forest, we saved the predictions in the same variable: `Y_pred`. Thus `Y_pred` currently contains the predictions of whichever analysis you ran most recently. If you went straight through the file, that would be the Random Forest model.
# 
# Thus to see the FPs and FNs for the Decision Tree model, we will first rerun the prediction using the Decision Tree model. If you would like to use the code below to see the FPs and FNs for a different model, change the first line below to store the predictions of a different model in Y_pred.

# %%
# rerun the label prediction (change the line below to specify which model to use)
Y_pred=dt_clf.predict(X_test)

# create new variables to hold the indices (i.e., row number) of all FPs and FNs
fp=[]
fn=[]

# go through all predictions to identify the errors and then determine whether each one is an FP or FN
for i in range (len(Y_test)):
  # identify FPs and store their indices
  if Y_pred[i] != Y_test[i] and Y_test[i] ==0:
       fp.append(i)
  # identify FNs and store their indices
  elif Y_pred[i] != Y_test[i] and Y_test[i]==1:
       fn.append(i)

# %% [markdown]
# ## Utilizing `RDkit` to get the molecule structure
# 
# We can use the `RDkit` library to display the structure of a molecule. To identify the molecule, we use the Simplified Molecular Input Line Entry System, or [SMILES](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system). SMILES is an unique chemical identifier in the form of an ASCII string. First we will go through the lists of FPs and FNs to get the SMILES strings and molecule names.
# 

# %%
# get the SMILES strings
fpmols=test.iloc[fp,0].values 
# get the molecule name
fpmols_name=test.iloc[fp,1].values 

# get the SMILES strings 
fnmols=test.iloc[fn,0].values 
# get the molecule names
fnmols_name=test.iloc[fn,1].values

# get lists of the FP and FN SMILES strings
fp_list = [Chem.MolFromSmiles(smiles) for smiles in fpmols]
fn_list = [Chem.MolFromSmiles(smiles) for smiles in fnmols]

# %%
# set molecule names for the FP plot displayed below
for i in range(len(fp_list)):
  mol=fp_list[i]
  mol.SetProp("_Name",fpmols_name[i])

# set molecule names for the FN plot displayed below
for i in range(len(fn_list)):
  mol=fn_list[i]
  mol.SetProp("_Name",fnmols_name[i])

# %% [markdown]
# Now let's display tables showing the index, name, and SMILES string of any FPs and FNs. Note that if the model performance was very good, there might not be any FPs and/or FNs, so the table(s) will be empty in that case.

# %%
# display a table showing the index, SMILES string, and name of all FPs
print('\033[1m' + 'FP List' + '\033[0m')
test.iloc[fp, 0:2]

# %%
# display a table showing the index, SMILES string, and name of all FNs
print('\033[1m' + 'FN List' + '\033[0m')
test.iloc[fn, 0:2]

# %% [markdown]
# **FP Group**

# %%
# display the molecular structures of all FPs
img1=Chem.Draw.MolsToGridImage(fp_list,
                               molsPerRow=4,
                               subImgSize=(200,200),
                               legends=[mol.GetProp('_Name') for mol in fp_list])
img1

# %% [markdown]
# **FN Group**

# %%
# display the molecular structures of all FNs
img2=Chem.Draw.MolsToGridImage(fn_list,
                               molsPerRow=4,
                               subImgSize=(200,200),
                               legends=[mol.GetProp('_Name') for mol in fn_list])
img2

# %% [markdown]
# ## Displaying FP or FN Spectrum
# 
# Edit the molecule index below to display the spectrum of a particular FP or FN for inspection. Do you see any spectral features that might explain the error?

# %%
# to display the spectrum of a FP
# in the line below, insert the index value of the molecule you want to see from the table above
fp_idx=78
# then display the figure
fp_mol=test.set_index('name').iloc[fp_idx,3:]
fig = go.Figure()
fig.add_trace(go.Scatter(x=fp_mol.index,
                         y=fp_mol,
                         name=fp_mol.name,
                         mode='markers'))
# Graph Layout
fig.update_layout(title='IR Spectra', title_x=0.5,
                  xaxis = dict(title='$Frequencies (cm^{-1})$'),
                  yaxis = dict(title='Normalized Intensity'))
fig.update_layout(showlegend=True)

# %%
# to display the spectrum of a FN
# in the line below, insert the index value of the molecule you want to see from the table above
fn_idx=137
# then display the figure
fn_mol=test.set_index('name').iloc[fn_idx,3:]
fig = go.Figure()
fig.add_trace(go.Scatter(x=fn_mol.index,
                         y=fn_mol,
                         name=fn_mol.name,
                         mode='markers'))

# Graph Layout
fig.update_layout(title='IR Spectra', title_x=0.5,
                  xaxis = dict(title='$Frequencies (cm^{-1})$'),
                  yaxis = dict(title='Normalized Intensity'))
fig.update_layout(showlegend=True)

# %% [markdown]
# # Conclusion
# We learned how to:
# - Use Google Colab/Vs Code Jupyter notebook environment
# - Load and preprocess data
# - Read IR spectra
# - Slice and index arrays with Pandas
# - Deal with bais in data and how to use sythetica data (cautiously)
# - Normalization and cut-off
# - Use train DecisionTree ML model with your data
# - Use predicit values on trained DecisionTree ML model
# - Measure accuracy of your model
# - Use RdK!t

# %% [markdown]
# # ------- Journal of Chemical Education Article --------
# If you would like to learn more about the decision tree and other simple machine-learning models, below is the link to the paper that helped me get started. Some suggestions on how to get the most out of this paper:
# 
# 1. To quickly start, read the .doc file from the SI, if you are comfortable with decision trees now
# 2. If you want to build chemical intuition, go through the main paper and then read the .doc file
# 3. The Jupyter Notebook they shared is slightly outdated, and most likely, it will give you errors if you try to run it. There are two solutions to this: [Ask me or go to the GitHub repository of the authors, https://github.com/elizabeththrall/MLforPChem/tree/main/MLforvibspectroscopy]
# 
# Reference:
# Machine Learning for Functional Group Identification in Vibrational Spectroscopy: A Pedagogical Lab for Undergraduate Chemistry Students, https://pubs.acs.org/doi/full/10.1021/acs.jchemed.1c00693
# 

# %% [markdown]
# 


