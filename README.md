# README: Women's E-commerce Clothing Reviews Sentiment Analysis and Complaint Clustering

This notebook performs a comprehensive analysis of women's e-commerce clothing reviews, aiming to understand customer sentiment and identify key areas for improvement based on negative feedback.

## Table of Contents

1.  [Project Overview](#project-overview)
2.  [Dataset](#dataset)
3.  [Data Loading and Initial Exploration](#data-loading-and-initial-exploration)
4.  [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5.  [Text Preprocessing](#text-preprocessing)
6.  [Sentiment Encoding](#sentiment-encoding)
7.  [Word Frequency Analysis](#word-frequency-analysis)
8.  [Model Training and Evaluation](#model-training-and-evaluation)
9.  [Streamlit Application and Deployment](#streamlit-application-and-deployment)
10. [Exported Data](#exported-data)

## 1. Project Overview

This project focuses on analyzing customer reviews for women's clothing items to:

*   Understand the distribution of ratings and recommendations.
*   Identify demographic trends (e.g., age groups) related to purchases.
*   Uncover the most urgent complaints from negative reviews.
*   Perform text preprocessing on review content.
*   Build and evaluate machine learning models (Logistic Regression, Random Forest) for sentiment classification.
*   Utilize K-Means clustering to categorize negative reviews into complaint groups.
*   Deploy an interactive Streamlit application for real-time review analysis.

## 2. Dataset

The dataset used is **'Womens Clothing E-Commerce Reviews'** from KaggleHub. It contains reviews with various attributes such as Clothing ID, Age, Title, Review Text, Rating, Recommended IND (indicator), Positive Feedback Count, Division Name, Department Name, and Class Name.

## 3. Data Loading and Initial Exploration

*   The dataset is downloaded using `kagglehub`.
*   Loaded into a pandas DataFrame.
*   Initial checks include `df.shape`, `df.info()`, `df.duplicated().sum()`, and `df.isnull().sum()` to understand its structure, data types, and missing values.

## 4. Exploratory Data Analysis (EDA)

*   **Urgent Complaints**: Negative reviews (Rating <= 2) are identified, and the top 10 most impactful complaints are extracted based on `Positive Feedback Count`.
*   **Age Distribution**: A histogram visualizes the distribution of reviewer ages.
*   **Rating Distribution**: Count plots show the frequency of each `Rating` and the relationship between `Rating` and `Recommended IND`.
*   **Department-wise Ratings**: A count plot visualizes ratings across different `Department Name` categories.
*   **Age Category Analysis**: A new `Categorical Age` column is created to group ages, and department preferences by age group are visualized.

## 5. Text Preprocessing

*   Rows with missing `Review Text` are dropped.
*   A `Number Words` column is created to store the count of words in each review.
*   NLTK libraries (`stopwords`, `WordNetLemmatizer`, `word_tokenize`) are downloaded and utilized.
*   Custom stopwords are defined, excluding negation words to preserve sentiment.
*   Helper functions are created for:
    *   `remove_stopwords`
    *   `remove_punctuation`
    *   `lemmatize`
    *   `clean_text` (combining lowercasing, regex for tags/URLs, punctuation removal, stopword removal, and lemmatization).
*   The `clean_text` function is applied to the `Review Text` to create a new `Cleaned Text` column.

## 6. Sentiment Encoding

A `rating_encoding` function is applied to create a binary `Rating Encoded` column:

*   `1` for Positive (Rating 4, 5, or Rating 3 with Recommended IND = 1)
*   `0` for Negative (Rating 1, 2, or Rating 3 with Recommended IND = 0)

## 7. Word Frequency Analysis

*   `CountVectorizer` is used to transform `Cleaned Text` into word frequency features.
*   The top 20 most frequent words are identified separately for positive and negative reviews and visualized using bar plots.

## 8. Model Training and Evaluation

*   **TF-IDF Feature Extraction**: `TfidfVectorizer` is used to convert `Cleaned Text` into TF-IDF features, limiting to `max_features=2000`.
*   **Feature Scaling**: `MaxAbsScaler` is applied to scale the features.
*   **Train-Test Split**: The data is split into training (80%) and testing (20%) sets.
*   **Model Training**:
    *   **Logistic Regression**: A `LogisticRegression` model is trained and its accuracy is reported.
    *   **Random Forest**: A `RandomForestClassifier` model is trained and its accuracy is reported.
*   **Evaluation**: A confusion matrix for the Logistic Regression model is visualized to understand its performance.
*   **K-Means Clustering**: `KMeans` is applied to the TF-IDF features of negative reviews to group similar complaints into 3 clusters.
*   **Saving Assets**: The trained `tfidf` vectorizer, `scaler`, `model` (Logistic Regression), and `kmeans` model are saved using `pickle` for future use in the Streamlit app.

## 9. Streamlit Application and Deployment

*   An `app.py` file is created using `%%writefile` to build a simple Streamlit application.
*   The app loads the saved models and allows users to input a review.
*   It predicts the sentiment (Positive/Negative) and, for negative reviews, assigns a complaint cluster number.
*   The application is deployed locally using `cloudflared` and `localtunnel`, providing a public URL for access.

## 10. Exported Data

*   The processed DataFrame `df` (including cleaned text, sentiment encoding, and categorical age) is saved to `cleaned_data.csv`.
