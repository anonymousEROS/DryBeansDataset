#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  7 20:16:41 2023

@author: blkcap
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
from sklearn.preprocessing import LabelEncoder



# =============================================================================
# import os
# for dirname, _, filenames in os.walk(''):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
# 
# =============================================================================
df = pd.read_excel('Dry_Bean_Dataset.xlsx')
df.head(10)
# =============================================================================
# df = pd.DataFrame(data)
# print(df)
# 
# df.shape
# dfSum= data.describe()
# print(dfSum)
# 
# df.info()
# 
# =============================================================================

# Select features to plot
features = ["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "AspectRation",
            "Eccentricity", "ConvexArea", "EquivDiameter", "Extent", "Solidity",
            "roundness", "Compactness", "ShapeFactor1", "ShapeFactor2", "ShapeFactor3",
            "ShapeFactor4"]

#Understanding the dataset
# Create subplots with multiple histograms
fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(15,15))

# Flatten the axs array for easy indexing
axs = axs.flatten()

# Create a histogram for each feature and add it to a subplot
for i, feature in enumerate(features):
    axs[i].hist(df[feature], bins=30)
    axs[i].set_title(feature)
    axs[i].set_xlabel(feature)
    axs[i].set_ylabel("Count")

# Remove extra subplots
while i+1 < len(axs):
    fig.delaxes(axs[i+1])
    i += 1

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

# Count occurrences of each category
class_counts = df["Class"].value_counts()
print(class_counts)

class_ = {'DERMASON':0, 'SIRA':1, 'SEKER':2, 'HOROZ':3, 'CALI':4, 'BARBUNYA':5, 'BOMBAY':6}
df['Class'] = df['Class'].replace(class_)

# Create a bar chart
class_counts.plot.bar()

# Set chart title and axis labels
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")

# Show the chart
plt.show()

# Since class is str replace with int
labelencoder = LabelEncoder()
df["Class"] = labelencoder.fit_transform(df['Class'])

df.head(10)
print(df)

# Compute correlation matrix
plt.figure(figsize=(12,12))
corr_matrix = df.corr()

# Create a heatmap of the correlation matrix
sns.heatmap(corr_matrix,annot = True, cmap=plt.cm.Reds)

# Set chart title
plt.title("Correlation Matrix")

# Show the chart
plt.show()


# =============================================================================
# 
# from sklearn.ensemble import RandomForestClassifier
# 
# Features = np.array(["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "AspectRation",
#             "Eccentricity", "ConvexArea", "EquivDiameter", "Extent", "Solidity",
#             "roundness", "Compactness", "ShapeFactor1", "ShapeFactor2", "ShapeFactor3",
#             "ShapeFactor4"])
# clf = RandomForestClassifier()
# clf.fit(df[Features], df['Class'])
# 
# importances = clf.feature_importances_
# sorted_idx = np.argsort(importances)
# 
# padding = np.arange(len(features)) + 0.5
# plt.barh(padding, importances[sorted_idx], align='center')
# plt.yticks(padding, features[sorted_idx])
# plt.xlabel("Relative Importance")
# plt.title("Variable Importance")
# plt.show()
# 
# =============================================================================








































# =============================================================================
# # Select features to plot
# features = ["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "AspectRation",
#             "Eccentricity", "ConvexArea", "EquivDiameter", "Extent", "Solidity",
#             "roundness", "Compactness", "ShapeFactor1", "ShapeFactor2", "ShapeFactor3",
#             "ShapeFactor4"]
# 
# # Create a histogram for each feature
# for feature in features:
#     plt.hist(df[feature], bins=30)
#     plt.title(feature)
#     plt.xlabel(feature)
#     plt.ylabel("Count")
#     plt.show()
# =============================================================================
# =============================================================================
# # =============================================================================
# # plt.hist(df['Area'], label = "Area")
# # =============================================================================
# plt.hist(df['Perimeter'], label = "Perimeter")
# # =============================================================================
# # plt.hist(df['MajorAxisLength'], label = "Major Axis Len")
# # =============================================================================
# plt.hist(df['MinorAxisLength'], label = "Minor Axis Len")
# plt.hist(df['AspectRation'], label = "AspectRation")
# plt.hist(df['Eccentricity'], label = "Eccentricity")
# # =============================================================================
# # plt.hist(df['ConvexArea'], label = "ConvexArea")
# # =============================================================================
# plt.hist(df['EquivDiameter'], label = "EquivDiameter")
# plt.hist(df['Extent'], label = "Extenet")
# plt.hist(df['Solidity'], label = "Solidity")
# plt.legend(loc = 'upper right')
# plt.show()
# 
# =============================================================================
