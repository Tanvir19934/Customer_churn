import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
import numpy as np
# Load the datasets
train_data = pd.read_csv('/Users/tanvirkaisar/Library/CloudStorage/OneDrive-UniversityofSouthernCalifornia/Semesters/Fall 2023/CSCI 561/customer_churn/Customer_Churn_Dataset/customer_churn_dataset-training-master.csv')

# Data preprocessing (to handle missing values and categorical encoding)
train_data.drop(columns=['CustomerID'], inplace=True)

########################################################################################################################################
# introducing some randomness
########################################################################################################################################

#flip_fraction = 0.8 
#indices = test_data.index
#num_to_flip = int(len(test_data) * flip_fraction)
#indices_to_flip = np.random.choice(indices, size=num_to_flip, replace=False)
#test_data.loc[indices_to_flip, 'Churn'] = test_data.loc[indices_to_flip, 'Churn'].apply(lambda x: 1 if x == 0 else 0)

# Introduce some significant randomness into the data
flip_fraction = 0.35  # Increase the fraction of elements flipped to 50%
std_dev = 0.2  # Increase the standard deviation of Gaussian noise

# Randomly select indices to flip 50% of the values
num_to_flip = int(len(train_data) * flip_fraction)
indices_to_flip = np.random.choice(train_data.index, size=num_to_flip, replace=False)

# Flip the selected values (0 -> 1, 1 -> 0)
train_data.loc[indices_to_flip, 'Churn'] = train_data.loc[indices_to_flip, 'Churn'].apply(lambda x: 1 if x == 0 else 0)

# Introduce stronger Gaussian noise to the binary data
train_data['Churn'] = train_data['Churn'] + np.random.normal(0, std_dev, size=train_data['Churn'].shape)

# Clip values to ensure they remain between 0 and 1
train_data['Churn'] = np.clip(train_data['Churn'], 0, 1)
train_data['Churn'] = train_data['Churn'].apply(lambda x: 1 if x > 0.55 else 0)

# Handle missing values using SimpleImputer (for numerical columns)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
train_data[['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']] = imputer.fit_transform(
    train_data[['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']])

# Convert categorical columns to numerical using Label Encoding
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
train_data['Gender'] = label_encoder.fit_transform(train_data['Gender'])
train_data['Subscription Type'] = label_encoder.fit_transform(train_data['Subscription Type'])
train_data['Contract Length'] = label_encoder.fit_transform(train_data['Contract Length'])

#####################################################################
# 1. Basic Exploratory Data Analysis (EDA)
#####################################################################

# Distribution of categorical variables
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.countplot(x='Gender', data=train_data)
plt.title('Gender Distribution')

plt.subplot(1, 3, 2)
sns.countplot(x='Subscription Type', data=train_data)
plt.title('Subscription Type Distribution')

plt.subplot(1, 3, 3)
sns.countplot(x='Contract Length', data=train_data)
plt.title('Contract Length Distribution')

plt.tight_layout()
plt.show()

# Distribution of numerical variables
plt.figure(figsize=(15, 10))
for i, column in enumerate(['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']):
    plt.subplot(3, 3, i+1)
    sns.histplot(train_data[column], kde=True)
    plt.title(f'{column} Distribution')
plt.tight_layout()
plt.show()

# Churn Distribution
plt.figure(figsize=(5, 5))
sns.countplot(x='Churn', data=train_data)
plt.title('Churn Distribution')
plt.show()

#####################################################################
# 2. Correlation Heatmap
#####################################################################

plt.figure(figsize=(12, 8))
correlation_matrix = train_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

#####################################################################
# 3. Churn Distribution Across Features
#####################################################################

# Create a figure with 3 subplots (1 row, 3 columns)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Churn by Subscription Type
sns.countplot(x='Subscription Type', hue='Churn', data=train_data, ax=axes[0])
axes[0].set_title('Churn by Subscription Type')

# Churn by Contract Length
sns.countplot(x='Contract Length', hue='Churn', data=train_data, ax=axes[1])
axes[1].set_title('Churn by Contract Length')

# Churn by Gender
sns.countplot(x='Gender', hue='Churn', data=train_data, ax=axes[2])
axes[2].set_title('Churn by Gender')

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()

#####################################################################
# 4. Pairplot for Numerical Features
#####################################################################

# Pairplot for selected numerical features
sns.pairplot(train_data[['Age', 'Tenure', 'Total Spend', 'Churn']], hue='Churn', palette='coolwarm', diag_kind = 'hist')
plt.show()
sns.pairplot(train_data[['Age', 'Payment Delay', 'Total Spend', 'Churn']], hue='Churn', palette='coolwarm')
plt.show()
#####################################################################
# 5. Boxplot for Outlier Detection
#####################################################################

# Boxplots for numerical features
plt.figure(figsize=(15, 10))
for i, column in enumerate(['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend']):
    plt.subplot(3, 3, i+1)
    sns.boxplot(x='Churn', y=column, data=train_data)
    plt.title(f'{column} Boxplot by Churn')
plt.tight_layout()
plt.show()

#####################################################################
# 6. Feature Importance (using Random Forest)
#####################################################################

train_data_2 = train_data.dropna(subset=['Churn'])

X_train = train_data_2.drop(columns=['Churn'])
y_train = train_data_2['Churn']

# Train Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Feature importance
importances = rf_model.feature_importances_
feature_names = X_train.columns
feat_importances = pd.Series(importances, index=feature_names)
feat_importances = feat_importances.sort_values()

# Plot Feature Importance
plt.figure(figsize=(10, 6))
feat_importances.plot(kind='barh')
plt.title('Random Forest Feature Importance')
plt.show()

#####################################################################
# Interactive Visualizations using Plotly
#####################################################################

# Churn by Age (interactive scatter plot)
fig = px.scatter(train_data, x='Age', y='Total Spend', color='Churn', 
                 title='Total Spend vs Age (Colored by Churn)')
fig.show()

2