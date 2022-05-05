# -*- coding: utf-8 -*-
"""
@author: Paula García Serrano

"""

# Read the libaries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# Read the normalized datasets
train = pd.read_csv('normalized_trainset.csv') # change the route to your train file
val = pd.read_csv('normalized_devset.csv') # change the route to your validation file

# Split the data
X = train['normalized_text'].copy()
y = train['Label'].copy()

X_test = val['normalized_text'].copy()
y_test = val['Label'].copy()

# Extract the features
transformer = TfidfVectorizer(ngram_range=(1,2))
transformer.fit(X)

X_transformed = transformer.transform(X)
X_test_transformed = transformer.transform(X_test)

# Create the model
model = RandomForestClassifier(random_state=10, n_jobs=-1)
model.fit(X_transformed, y)

# Predict using the model
predictions_test = model.predict(X_test_transformed)

# Save the model
dump(model, 'TFIDF_ngrams_RandomForestClassifier.joblib')