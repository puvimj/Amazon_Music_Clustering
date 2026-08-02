# 🎵 Amazon Music Clustering – Music Recommendation & Analytics

## 📌 Project Overview

This project uses **Unsupervised Machine Learning** to analyze Amazon Music tracks and group songs into meaningful clusters based on their audio characteristics.

The objective is to discover natural patterns in music without using predefined labels.

The project includes:

- Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing
- Feature selection
- Outlier handling
- Feature scaling
- PCA for dimensionality reduction
- K-Means clustering
- Cluster evaluation
- Cluster profiling and interpretation
- Interactive Streamlit dashboard
- Export of the final clustered dataset

---

## 🎯 Problem Statement

Music platforms contain a large number of songs with different audio characteristics.

The objective of this project is to:

> **Group songs with similar audio characteristics into meaningful clusters using unsupervised machine learning.**

These clusters can help understand different types of music and can potentially support music recommendation and content categorization.

---

## 📂 Dataset

The dataset contains approximately **95,837 songs** and includes information related to:

### 🎧 Audio Features

- Danceability
- Energy
- Loudness
- Speechiness
- Acousticness
- Instrumentalness
- Liveness
- Valence
- Tempo
- Duration

### 📊 Other Features

- Song popularity
- Artist popularity
- Artist followers
- Explicit indicator
- Release year
- Key
- Mode
- Time signature

---

## 🔍 Exploratory Data Analysis

EDA was performed to understand:

- Dataset structure
- Missing values
- Duplicate records
- Data types
- Feature distributions
- Outliers
- Correlations
- Audio feature relationships
- Feature distributions across clusters

Visualizations generated during EDA include:

- Distribution plots
- Box plots
- Correlation heatmaps
- Cluster comparison charts
- Standardized cluster heatmaps
- Audio feature distributions by cluster
- Cluster size visualization

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

1. Removed unwanted columns.
2. Checked for duplicate records.
3. Handled missing values.
4. Converted release date information into `release_year`.
5. Checked distributions of important numerical features.
6. Identified and handled outliers where appropriate.
7. Selected relevant audio features for clustering.
8. Standardized numerical features using `StandardScaler`.

---

## 🎧 Features Used for Clustering

The primary audio features used for clustering were:

```text
danceability
energy
loudness
speechiness
acousticness
instrumentalness
liveness
valence
tempo
duration_ms
