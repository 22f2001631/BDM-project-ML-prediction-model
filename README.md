# VMTC - Profit Prediction System

## Overview

This  builds a simple machine learning pipeline using operational data from Vijay Motor Transport Company. The model is trained on historical trip data to predict expected Revenue, Expense, and Profit based on Route, Day, and Phase.

## Dataset

The dataset contains digitized daily logs with the following key fields:

* Route ID
* Day
* Revenue components (To, Fro, Luggage)
* Expense components (Diesel, Wages, Maintenance, etc.)
* Total Revenue, Total Expense, Profit

Dataset link: (Records.csv)[https://docs.google.com/spreadsheets/d/16PePWCdJys-Zz9EmaHHuvm7zPA7yRFZFIF1iSynUzx8/edit?gid=0#gid=0]

## Files

* train.py: trains the model and saves model.pkl and encoders.pkl
* predict.py: loads the saved model and makes predictions
* //data/records.csv: input dataset
* requirements.txt: dependencies

## How to Run

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Place the dataset as:

```
data.csv
```

3. Train the model:

```
python train.py
```

This will generate:

* model.pkl
* encoders.pkl

4. Run prediction:

```
python predict.py
```

## Input Format

You will be prompted to enter:

* Route ID (example: KRW-SRG)
* Day (example: Sunday)
* Phase (Phase1 / Phase2 / Phase3)

## Sample Input

```
KRW-SRG
Sunday
Phase2
```

## Sample Output

```
Predicted Revenue: ₹XXXX
Predicted Expense: ₹XXXX
Predicted Profit: ₹XXXX
```

## Notes

* Input values must match training data categories
* Model can be retrained by updating the dataset and running train.py again
* Predictions are approximate and depend on historical patterns
