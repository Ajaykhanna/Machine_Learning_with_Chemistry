# Predicting Vertical Excitation Energies (VEEs) of Propanol Molecule Using Random Forest 🚀

## Overview 📚

This project aims to predict the vertical excitation energies (VEEs) of the propanol molecule using a Random Forest regression model. The data for this project was synthesized through molecular dynamics (MD) simulations and quantum mechanical calculations.

## Data Synthesis 🧬

- **MD Simulation:**
  - Performed using BCC charge parameterized GAFF2 forcefield for propanol in the gas phase in AMBER20.
  - 4000 uncorrelated snapshots were obtained in the NVT ensemble.
  - The trajectory was converted to XYZ format, and individual frames were extracted using utility scripts.

- **Quantum Mechanical Calculations:**
  - VEEs on 4000 snapshots were computed on the unoptimized structures using TDA/CAM-B3LYP/6-31G(d) method with TeraChem.

## Machine Learning Model 🤖

- **Model Type:** Random Forest regression model.
- **Input Features:** 3D coordinates represented as the Coulomb matrix.
- **Target Variable:** VEEs.
- **Training Data:** 80% of the dataset.
- **Testing Data:** 20% of the dataset.

## Model Performance 📈

- **Mean Absolute Error (MAE):** 0.07
- **Root Mean Squared Error (RMSE):** 0.08
- **Coefficient of Determination (R²):** 0.72

## Conclusion 🏆

This project demonstrates the effective use of Random Forest regression to predict VEEs of the propanol molecule, leveraging MD simulation data and quantum mechanical calculations. The model exhibits strong predictive performance, with low error metrics and a respectable R² value.
