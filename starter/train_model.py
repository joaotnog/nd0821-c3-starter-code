# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from starter.ml.data import process_data
from starter.ml.model import *
import pandas as pd
import joblib

# Add the necessary imports for the starter code.
data = pd.read_csv('./data_mod/census_mod.csv')

# Add code to load in the data.

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, 
    encoder = encoder, lb = lb, label="salary", training=False
)


# Train and save a model.
model = train_model(X_train, y_train)
joblib.dump(model, 'starter/model/model.pkl')
joblib.dump(encoder, 'starter/model/encoder.pkl')
joblib.dump(lb, 'starter/model/lb.pkl')

# Compute metrics on test 
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
dict_test_metrics = dict(precision = precision,
                         recall = recall,
                         fbeta = fbeta)
joblib.dump(dict_test_metrics, './model/dict_test_metrics.pkl')
