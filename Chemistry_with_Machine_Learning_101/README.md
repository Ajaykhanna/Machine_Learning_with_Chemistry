Machine learning can be a difficult concept to understand, especially if you're not familiar with how it can be used to predict chemical properties. For chemists, this can be even more challenging since there aren't any courses that introduce the use of data analytics tools, let alone machine learning programs. That's why I created this GitHub repository to help chemists (like myself) understand the basics of machine learning and its applications in predicting properties like toxicity, solubility, IR frequencies, and hybridization. My aim is to simplify the ML concepts and relate them to chemical intuition, so it's easier to understand and apply.
This repo introduces students to the use of machine leaning in chemistry and they can benefit in accelerating our understanding of vast chemical space.

## 🧪 [Identify carbonyl groups like a machine!](https://github.com/Ajaykhanna/Machine_Learning_with_Chemistry/blob/main/Chemistry_with_Machine_Learning_101/Carbonyl_Binary_Classification_UCM_Khanna_Ajay.ipynb) : 

Machine learning allows computers to make predictions and decisions by learning from data, without being explicitly programmed for the task. As shown in this code, a decision tree model can be trained on infrared spectral data to predict whether molecules contain a specific functional group like a carbonyl. After preprocessing the data, the model learns decision rules that discriminate between the two classes. Its performance is then evaluated on a test set to determine the model's ability to generalize.

This demonstrates how machine learning can be applied in chemistry to analyze spectral data and identify patterns that correlate with properties like functional group presence. Rather than relying solely on human-derived rules, machine learning algorithms can determine their own correlations. This allows large volumes of chemical data to be rapidly analyzed to discover new insights. Machine learning is becoming widely used in chemistry for spectroscopic analysis, molecular property prediction, reaction optimization, and drug discovery. By training models on experimental data, it is possible to predict chemical behaviors and guide experiments without extensive lab work.

This code trains a decision tree model on IR spectra to predict whether molecules contain a carbonyl functional group. It demonstrates key steps in applying supervised ML to chemistry:
- 📥 Loading and preprocessing training + test data
- 🌳 Training a decision tree classifier 
- ✅ Testing on holdout data & analyzing model performance
- 🔍 Investigating errors by visualizing molecules & spectra

See molecules through the eyes of a machine! Trained on real experimental data, this model can predict the presence of specific chemical functional groups - no wet lab required!