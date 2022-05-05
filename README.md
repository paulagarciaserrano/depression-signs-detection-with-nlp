# Depression Signs Detection with NLP
An Artificial Intelligence Approach to Detect Depression Signs in Social Media Text

## Introduction
I have explored how AI algorithms can be applied to detect signs of depression from social media text in English. 

To do so, I have trained my models on a very recent corpus, extracted from a Codalab competition: <a href="https://competitions.codalab.org/competitions/36410">Detecting Signs of Depression from Social Media Text-LT-EDI@ACL 2022</a>. 

The experiments have covered classical machine learning, deep learning and transfer learning algorithms to solve the task. After comparing the results among the three categories, the best model in terms metric preformance turned out to be the twitter-roberta-base-mar2022, with a score of 0.54, and in terms of latency and the explainability, the random forest with a macro f1-score of 0.52. Because when speaking about mental health disorders detection, researchers need to be able to offer transparency and explain how the prediction turned out to be what it was, I have chosen the random forest model as the best to solve the task at hand. 

Furthermore, a website has been created with this best model, so that any individual can test the predictions over the desired text, regardless of their programming capabilities. The website can be found on: https://paulagarciaserrano.pythonanywhere.com/.

## Content

This respository contains the files used to develop the aforementioned model.

### 1. Dataset

The competition organizers offer three datasets: train, development and test. However, as the _Label_ feature is only present in two of the three datasets, only the train and development sets can be used to build the predictive model. As the performance of the models could not be evaluated in the proposed test set, the development set acted as the test set, and the train set was divided into train and development itself, following a 70-30 stratified ratio.

The dataset can be extracted from: https://competitions.codalab.org/competitions/36410#learn_the_details.

#### 1.1. Preprocessing

When building the competition datasets, the researchers performed a basic cleaning of the data by removing non-ASCII characters and emoticons. After this, to fill the missing values, they combined the “title” and “text” columns into a single “text data” column (Kayalvizhi et al., 2022).

As the datasets come from social media, a more profound cleaning was performed: all the text was converted to lowercase, the full expressions substituting abbreviations (e.g., don't → do not, it's → it is, he'll → he will) and unwanted characters were removed, including tags or mentions (e.g., @name), hashtags, weblinks, remaining emojis, punctuation, trailing whitespaces and stop words. Furthermore, all the text was tokenized and stemmed using Porter stemming.

The ``01_Data_Preprocessing.py`` file contains the code used to perform this cleaning.

### 2. Model

The following sections have been included in the ``02_Modeling.py`` file.

#### 2.1. Feature Extraction

After having the data cleaned, numerical features need to be extracted out of the raw text for the models to understand this text data. For the case of the random forest, the best vectorizer turned out to be the TF-IDF with N-grams.

#### 2.2. Model Creation

A random forest combines many decision trees to generate the final prediction by bootstrap aggregation and bagging. In this implementation, the best configuration was:

* Nº estimators: 100
* Criterion: Gini
* Minimum samples split: 2
* Minimum sample leaf: 1
* Minimum weight fraction leaf: 0
* Maximum features: Auto
* Minimum impurity decrease: 0
* Random split: 10
* Bootstrap: True
* Out-of-bag samples: False
* Warm start: False

The trained model is uploaded to the repository under the name: ``TFIDF_ngrams_RandomForestClassifier.joblib``

## References

Kayalvizhi, S., & Thenmozhi, D. (2022). Data set creation and empirical analysis for detecting signs of depression from social media postings. _arXiv preprint arXiv:2202.03047_.
