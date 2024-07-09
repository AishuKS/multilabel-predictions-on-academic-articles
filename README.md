# Multi-Label Predictions on Academic Articles
## Project Overview
This repository contains the code and resources for the project "Multi-Label Predictions on Academic Articles". The primary objective of this project is to classify research papers into multiple labels (e.g., Computer Science, Physics, Mathematics, etc.) using Natural Language Processing (NLP) techniques. The project explores various models and algorithms to achieve robust multi-label classification.

## Team Members
Aishwarya Kandasamy  
Deepika Alagiriswamy Panneerselvam  
Mubariz Ahmed Khan  
Priyanka Basani  
Ranjitha Aswath  

## Table of Contents
1. Introduction
2. Objectives
3. Dataset
4. Data Preprocessing
5. Models and Approaches
	- Approach 1: Machine Learning Models
	- Approach 2: BERT Model
	- Approach 3: Neural Networks
	- Approach 4: LLM - Mistral 7B
	- Approach 5: Rakel Algorithm
6. Conclusions
7. Project Challenges
8. Future Work
9. References

## Introduction
In today's vast landscape of academic publications, the ability to efficiently categorize and classify research papers based on their content is crucial for knowledge retrieval and domain-specific insights. The main focus of this research is on using NLP and multi-label classification approaches to categorize research papers.

## Objectives
1. Multi-label Classification with NLP: Implement a robust multi-label classification system for research papers using NLP techniques.
2. Knowledge Retrieval Enhancement: Improve knowledge retrieval by efficiently categorizing research papers using NLP to enable effective information retrieval for researchers.

## Dataset
The dataset used in this project consists of research articles categorized into six labels:
1. Quantitative Finance
2. Quantitative Biology
3. Computer Science
4. Physics
5. Mathematics
6. Statistics
- Training Data: 20,972 samples
- Testing Data: 8,989 samples

## Data Preprocessing
- Merged "Title" and "Abstract" as a "Text" column.
- Cleaned and standardized data (removed HTML tags, punctuation, special characters, stopwords, etc.).
- Implemented Dynamic MLSMOTE to address class imbalance.
- Added features like Text Length, Text Word Count, Average Word Length, and Contains Numerals.

## Models and Approaches
## Approach 1: Machine Learning Models
1. Vectorization Methods:
- TF-IDF
- Word2Vec

2. Transformation Methods:
- Binary Relevance (BR)
- Label Powerset (LP)
- Classifier Chains (CC)

3. Machine Learning Algorithms:
- Random Forest
- Logistic Regression
- Support Vector Machine (SVM)

## Approach 2: BERT Model
- Used the BERT model with custom datasets and tokenizers.
- Utilized "bert-base-uncased" pre-trained model from the Hugging Face library.
- Implemented training and validation loops, checkpointing, and monitoring.

## Approach 3: Neural Networks
1. Using Word2Vec:
- Tokenization and padding sequences.
- Defined a neural network model using Keras.
- Evaluated the model using various metrics.

2. Using TF-IDF:
- TF-IDF vectorization.
- Defined a neural network model using Keras.
- Evaluated the model using various metrics.

## Approach 4: LLM - Mistral 7B
- Explored the Mistral 7B model for multi-label classification.
- Evaluated the model based on several benchmarks and metrics.

## Approach 5: Rakel Algorithm
- Implemented the Rakel Algorithm for ensemble learning.
- Used RandomForestClassifier as the base classifier with TF-IDF vectorization.

## Conclusions
- BERT outperformed other methods and algorithms.
- TF-IDF was generally more effective than Word2Vec.
- Balanced data outperformed imbalanced data in every model.

## Project Challenges
- Addressing class imbalance.
- Limited GPU resources prevented the use of high-performance LLMs like RoBERTa and GPT.
- Extended runtime with some transformation methods and algorithms.

## Future Work
- Explore hybrid approaches combining MLC methods and classification algorithms.
- Use advanced text representation techniques like RoBERTa and XLNet.
- Implement adaptive imbalance handling methods.
- Experiment with ensemble learning strategies for enhanced model robustness.
