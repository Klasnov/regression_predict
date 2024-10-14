# Singapore Car Resale Price Prediction

This is the course project of CS5228, School of Computing, National University of Singapore.



## Overview

In this project, we look into the market for used cars in Singapore. Car ownership in Singapore is rather expensive which includes the very high prices for new and used cars (compared to many other countries). There are many stakeholders in this market. Buyers and sellers want to find good prices, so they need to understand what affects the value of a car. Online platforms facilitating the sale of used cars, on the other hand, want to maximize the number of sales/transactions.

The goal of this task is to predict the resale price of a car based on its properties (e.g., make, model, mileage, age, power, etc). It is therefore first and foremost a regression task. These different types of information allow you to come up with features for training a regressor. It is part of the project for you to justify, derive and evaluate different features. Besides predicting the outcome in terms of a dollar value, other useful results include the importance of different attributes, the evaluation and comparison of different regression techniques, an error analysis and discussion about limitations and potential extensions, etc.



## Evaluation

The evaluation metric for this competition is Root Mean Squared Error (RSME). The RSME is a common metric to evaluate regression tasks. We use the RSME (instead of the Mean Squared Error) so that the error values have the correct unit, which is SGD for this task.
