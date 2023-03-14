# Hotel_Reviews_NLP_Analysis

<img src="https://www.reviewtrackers.com/wp-content/uploads/Hotel-Reviews.jpg">

In the first part, I will begin with some Exploratory Data Analysis (EDA), and then move into data augmentation, modelling, and iteration over model improvements.

First let's look at the dataset, which can be downloaded [here](https://api.brainstation.io/content/link/1ZaOufpJjCLzUS8VaUnrgvjiupiwqWdC_).

The objective of this part is to analyze a dataset of hotel reviews and predict whether a review is 'good' or 'not good' based on its Reviewer_Score. The project will involve data wrangling, feature engineering, and building a classification model using count vectorization.

Exploratory Data Analysis:
The first step is to explore the dataset, starting with the shape of the dataset. All reviews are given as decimal values, so we need to convert them into integers ranging from 1 to 10. We will also investigate the distribution of Reviewer_Score, both expected and actual. We will identify which columns in the dataset are numeric and which are non-numeric. We will also look for any potential problems with the distribution that may affect our classification model.

Data Wrangling:
We will build the proper dataset separation by subsampling ~10% of the data to ensure the dataset runs efficiently. The Reviewer_Score column will be converted into a binary column, with scores below 9 being encoded as 0 ('not good') and scores 9 and 10 as 1 ('good'). We will also convert the identified non-numeric columns to numeric, drop non-numeric columns except Positive_Review and Negative_Review, and split the data into train and test sets.

Feature Engineering:
We will use count vectorization to combine Positive_Review and Negative_Review with the numeric data. The count vectorizer will separately vectorize each column, resulting in two sparse matrixes, which will then be combined with the numeric data. We may need to adjust the min_df parameter to optimize the vectorization process.

Conclusion:
The min_df parameter in count vectorization determines the minimum frequency of a word in the document required to be included in the sparse matrix. This project will provide insight into the distribution of hotel reviews and identify features that affect the Reviewer_Score. The classification model will be built using count vectorization and trained on the train dataset to predict Reviewer_Score on the test dataset.

---

<img src="https://www.webintravel.com/wp-content/uploads/2019/04/GettyImages-802970402.jpg">

In part 2, I will develop several machine learning models to correctly label the sentiment behind hotel reviews.

You have been provided with a cleaned and preprocessed dataset you must use, which differs from that for Part 1. [Download the data here](https://api.brainstation.io/content/link/16DkHhup_0nI5LgZzYsdfsfwN60DKaiAN).

In this part, the target column is the "rating" column which is a binary column denoting good ratings as 1 and bad ones as 0. 

The aim of part2 is to explore different machine learning algorithms and techniques on a given dataset to model the relationship between a set of independent variables and a dependent variable. The dataset consists of reviews of a product, and the goal is to predict whether a review is positive or negative based on the text content.

1. Logistic Regression:
We will employ a linear classifier on the dataset by fitting a logistic regression model with the solver set to lbfgs. We will evaluate the accuracy score on the test set and identify the 20 words most predictive of a good or bad review using regression coefficients. Additionally, we will reduce the dimensionality of the dataset using PCA and examine the relationship between the number of dimensions and run-time for logistic regression. Finally, we will list one advantage and one disadvantage of dimensionality reduction.

2. K-Nearest Neighbour:
We will employ a K-Nearest Neighbour classifier on the dataset by fitting a KNN model and evaluate the accuracy score on the test set. Since KNN is a computationally expensive model, we will explore reducing the number of observations in the dataset and examine the relationship between the number of observations and run-time for KNN. We will also list one advantage and one disadvantage of reducing the number of observations. Finally, we will use the dataset to find an optimal value for K in the KNN algorithm by splitting the dataset into train and validation sets.

3. Decision Tree:
We will employ a Decision Tree classifier on the dataset by fitting a decision tree model and evaluate the accuracy score on the test set. Additionally, we will use the data set (or a subsample) to find an optimal value for the maximum depth of the decision tree by splitting the dataset into train and validation sets. We will provide two advantages of decision trees over KNN and two weaknesses of decision trees (classification or regression trees). Finally, we will explain the purpose of the validation set and how it differs from the test set.

4. Re-run Logistic Regression or Decision Tree:
We will re-run the logistic regression or decision tree on the dataset by performing a 5-fold cross-validation to optimize the hyperparameters of the model. We will examine the confusion matrix of the best model on the test set. Finally, we will create one new feature and explain how it can improve accuracy. We will run the model again and evaluate whether the accuracy score of the best model has improved on the test set after adding the new feature.

Overall, this project will provide a comprehensive understanding of different machine learning algorithms and techniques and how they can be applied to a real-world dataset to solve a classification problem.
