# MSPE 49600 — Project 03

**Name:** Andrew Ramthun  
**Course:** MSPE 49600  
**Project:** Project 03  

## Project Description

This project applies predictive modeling techniques to motorsports data using Python. The first portion models mean Sector 3 lap time as a function of rear-wing angle and rear ride height using Gaussian radial basis functions (RBFs) and a quadratic reference model. Model performance is evaluated using leave-one-out cross-validation and an independent test set.

The second portion uses simulated lap-time data to compare neural-network models with ReLU and tanh activation functions against a quadratic reference model. Fuel load and tire age are used as inputs to predict lap time, and the models are evaluated using training and test error metrics.

## Methods Used

- Data cleaning and preprocessing
- Train/test splitting
- Input normalization
- Gaussian radial basis functions
- Leave-one-out cross-validation
- Hyperparameter grid search
- Quadratic regression
- Neural networks using `MLPRegressor`
- Model validation and visualization

## Main Python Packages

- NumPy
- Pandas
- Matplotlib
- SciPy
- Scikit-learn

## Repository Contents

The repository contains the project notebook, required CSV datasets, generated output CSV files, figures, `requirements.txt`, `.gitignore`, and this README file.