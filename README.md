# AI-Powered LMS – Data Science 

## 1 | Project Overview
This project involves building a personalized course recommendation engine as part of an AI-powered Learning Management System (LMS). The recommendation engine uses sentence-embedding models to suggest courses to users based on their interests, previous courses, and learning goals.

## 2 | Embedding Models Evaluated
| Model name | Mean Precision | Mean Recall | Mean F1-Score | Mean Average Precision (MAP) | ROC AUC |
|------------|--------------------|--------|-------|--------|----------|
| all-MiniLM-L6-v2 | 0.467 | 0.833 | 0.548 | 0.881 | 0.855
| all-MPNet-base-v2 | 0.400 | 0.783 | 0.490 | 0.875 | 0.883
| paraphrase-multilingual-MiniLM-L12-v2 | 0.400 | 0.767 | 1.481 | 0.871 | 0.848

## 3 | How the recommendation should work
1. User inputs their preferences: interest, past work experience, career goal, etc
2. Encode using an embedding model
3. Compute cosine similarity with course embeddings
4. Rank and return top-k results
5. Serve via fast API


