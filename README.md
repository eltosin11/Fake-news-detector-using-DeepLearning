# Fake news detector using NLP and DeepLearning

## The dataset
News headlines dataset for sarcasm detection is collected from two news websites. TheOnion, which aims at producing sarcastic versions of current news. Real (and non-sarcastic) news headlines are collected from HuffPost.

Each record consists of three attributes:
-- is_sarcastic: 1 if the record is sarcastic otherwise 0
-- headline: the headline of the news article
-- article_link: link to the original news article. Useful in collecting supplementary data

## Key objectives
-- To build a basic Bag of Words Term Frequency-Inverse Document Frequency (TF-IDF) + Logistic Regression baseline model 

-- Try different deep learning + NLP methods like pre-trained embeddings + DL models, Universal Embeddings, Transformers etc and try to build the most accurate model and break the baseline score

-- Showcase model performance on test data using confusion matrix and classification reports
## Results 
-- 92% accuracy score was obtained using the DistillBERT model (the highest of Logistic, FastTest Embeddings + CNN; Neural Network Language Model and BERT models) 
