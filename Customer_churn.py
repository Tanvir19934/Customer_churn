import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np


# Load the datasets
train_data = pd.read_csv('/Users/tanvirkaisar/Library/CloudStorage/OneDrive-UniversityofSouthernCalifornia/Semesters/Fall 2023/CSCI 561/customer_churn/Customer_Churn_Dataset/customer_churn_dataset-training-master.csv')
test_data = pd.read_csv('/Users/tanvirkaisar/Library/CloudStorage/OneDrive-UniversityofSouthernCalifornia/Semesters/Fall 2023/CSCI 561/customer_churn/Customer_Churn_Dataset/customer_churn_dataset-testing-master.csv')


########################################################################################################################################
# introducing some randomness
########################################################################################################################################

#flip_fraction = 0.8 
#indices = test_data.index
#num_to_flip = int(len(test_data) * flip_fraction)
#indices_to_flip = np.random.choice(indices, size=num_to_flip, replace=False)
#test_data.loc[indices_to_flip, 'Churn'] = test_data.loc[indices_to_flip, 'Churn'].apply(lambda x: 1 if x == 0 else 0)

# Introduce some significant randomness into the data
flip_fraction = 0.1  # Increase the fraction of elements flipped to 50%
std_dev = 0.1  # Increase the standard deviation of Gaussian noise

# Randomly select indices to flip 50% of the values
num_to_flip = int(len(train_data) * flip_fraction)
indices_to_flip = np.random.choice(train_data.index, size=num_to_flip, replace=False)

# Flip the selected values (0 -> 1, 1 -> 0)
train_data.loc[indices_to_flip, 'Churn'] = train_data.loc[indices_to_flip, 'Churn'].apply(lambda x: 1 if x == 0 else 0)

# Introduce stronger Gaussian noise to the binary data
train_data['Churn'] = train_data['Churn'] + np.random.normal(0, std_dev, size=train_data['Churn'].shape)

# Clip values to ensure they remain between 0 and 1
train_data['Churn'] = np.clip(train_data['Churn'], 0, 1)
train_data['Churn'] = train_data['Churn'].apply(lambda x: 1 if x > 0.5 else 0)




# Data preprocessing
train_data.drop(columns=['CustomerID'], inplace=True)
test_data.drop(columns=['CustomerID'], inplace=True)

# Handle missing values using SimpleImputer (for numerical columns)
imputer = SimpleImputer(strategy='mean')
train_data[['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']] = imputer.fit_transform(
    train_data[['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']])
test_data[['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']] = imputer.transform(
    test_data[['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']])

# Convert categorical columns to numerical using Label Encoding
label_encoder = LabelEncoder()
train_data['Gender'] = label_encoder.fit_transform(train_data['Gender'])
test_data['Gender'] = label_encoder.transform(test_data['Gender'])

train_data['Subscription Type'] = label_encoder.fit_transform(train_data['Subscription Type'])
test_data['Subscription Type'] = label_encoder.transform(test_data['Subscription Type'])

train_data['Contract Length'] = label_encoder.fit_transform(train_data['Contract Length'])
test_data['Contract Length'] = label_encoder.transform(test_data['Contract Length'])

# Separate features and target variable
X_train = train_data.drop(columns=['Churn'])
y_train = train_data['Churn']
X_test = test_data.drop(columns=['Churn'])
y_test = test_data['Churn']

# Standardize/Scale the numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Split the data into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

# Remove rows with missing target values
X_train_clean, y_train_clean = X_train_split[~pd.isnull(y_train_split)], y_train_split[~pd.isnull(y_train_split)]

####################################################################
# Logistic Regression Model
####################################################################
logistic_model = LogisticRegression()
logistic_model.fit(X_train_clean, y_train_clean)
y_pred_logistic = logistic_model.predict(X_val_split)
y_pred_logistic_proba = logistic_model.predict_proba(X_val_split)[:, 1]

####################################################################
# Random Forest Model
####################################################################
rf_model = RandomForestClassifier()
rf_model.fit(X_train_clean, y_train_clean)
y_pred_rf = rf_model.predict(X_val_split)
y_pred_rf_proba = rf_model.predict_proba(X_val_split)[:, 1]

########################################################################################################################################
# XGBOOST Model
########################################################################################################################################
# Initialize the XGBoost classifier
base_model = xgb.XGBClassifier(objective='binary:logistic', n_jobs=-1, eval_metric='auc')

# Define parameter grid for GridSearch
test_par = {
    'max_depth': [2, 10, 20],
    'learning_rate': [0.1, 0.2, 0.5],
    'n_estimators': [5, 10, 20],
    'min_child_weight': [0.05, 0.15]
}

# Using StratifiedKFold for cross-validation
cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Grid Search with cross-validation
XGB_gsearch = GridSearchCV(base_model, param_grid=test_par, scoring='roc_auc', cv=cv_splitter, verbose=1, n_jobs=-1, refit=True)
XGB_gsearch.fit(X_train_clean, y_train_clean)

# Get results and best parameters
XGB_result_df = pd.DataFrame(XGB_gsearch.cv_results_)
print(f"Best ROC-AUC Score: {XGB_gsearch.best_score_}")
print(f"Best Parameters: {XGB_gsearch.best_params_}")

# Make predictions on the validation set
y_pred_xgb = XGB_gsearch.best_estimator_.predict(X_val_split)
y_pred_xgb_proba = XGB_gsearch.best_estimator_.predict_proba(X_val_split)[:, 1]  # Probabilities for ROC AUC

# Feature importance
X_feature = X_train.columns  # Assuming X_train is the original feature DataFrame
RF_feat = pd.DataFrame(data=XGB_gsearch.best_estimator_.feature_importances_, index=X_feature, columns=['importance'])

# Plot feature importance
RF_feat.sort_values('importance').plot(kind='barh', title='XGB Feature Importance')
plt.show()

####################################################################
# Evaluate all models and Display evaluation metrics
####################################################################
def evaluate_model(y_true, y_pred, y_pred_proba):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    return accuracy, precision, recall, f1, roc_auc

logistic_eval = evaluate_model(y_val_split, y_pred_logistic, y_pred_logistic_proba)
rf_eval = evaluate_model(y_val_split, y_pred_rf, y_pred_rf_proba)
xgb_eval = evaluate_model(y_val_split, y_pred_xgb, y_pred_xgb_proba)

print("Logistic Regression Performance:")
print(f"Accuracy: {logistic_eval[0]:.4f}, Precision: {logistic_eval[1]:.4f}, Recall: {logistic_eval[2]:.4f}, F1 Score: {logistic_eval[3]:.4f}, ROC-AUC: {logistic_eval[4]:.4f}")

print("\nRandom Forest Performance:")
print(f"Accuracy: {rf_eval[0]:.4f}, Precision: {rf_eval[1]:.4f}, Recall: {rf_eval[2]:.4f}, F1 Score: {rf_eval[3]:.4f}, ROC-AUC: {rf_eval[4]:.4f}")

print("\nXGBoost Performance:")
print(f"Accuracy: {xgb_eval[0]:.4f}, Precision: {xgb_eval[1]:.4f}, Recall: {xgb_eval[2]:.4f}, F1 Score: {xgb_eval[3]:.4f}, ROC-AUC: {xgb_eval[4]:.4f}")



import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

# Load the datasets
train_data = pd.read_csv('/Users/tanvirkaisar/Library/CloudStorage/OneDrive-UniversityofSouthernCalifornia/Semesters/Fall 2023/CSCI 561/customer_churn/Customer_Churn_Dataset/customer_churn_dataset-training-master.csv')
test_data = pd.read_csv('/Users/tanvirkaisar/Library/CloudStorage/OneDrive-UniversityofSouthernCalifornia/Semesters/Fall 2023/CSCI 561/customer_churn/Customer_Churn_Dataset/customer_churn_dataset-testing-master.csv')

# Data preprocessing
train_data.drop(columns=['CustomerID'], inplace=True)
test_data.drop(columns=['CustomerID'], inplace=True)

# Handle missing values using SimpleImputer (for numerical columns)
imputer = SimpleImputer(strategy='mean')
train_data[['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']] = imputer.fit_transform(
    train_data[['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']])
test_data[['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']] = imputer.transform(
    test_data[['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']])

# Convert categorical columns to numerical using Label Encoding
label_encoder = LabelEncoder()
train_data['Gender'] = label_encoder.fit_transform(train_data['Gender'])
test_data['Gender'] = label_encoder.transform(test_data['Gender'])

train_data['Subscription Type'] = label_encoder.fit_transform(train_data['Subscription Type'])
test_data['Subscription Type'] = label_encoder.transform(test_data['Subscription Type'])

train_data['Contract Length'] = label_encoder.fit_transform(train_data['Contract Length'])
test_data['Contract Length'] = label_encoder.transform(test_data['Contract Length'])

# Separate features and target variable
X_train = train_data.drop(columns=['Churn'])
y_train = train_data['Churn']
X_test = test_data.drop(columns=['Churn'])
y_test = test_data['Churn']

# Standardize/Scale the numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Split the data into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

########################################################################################################################################
# XGBOOST Model with Cross-Validated ROC Curve Plot
########################################################################################################################################
# Initialize the XGBoost classifier
base_model = xgb.XGBClassifier(objective='binary:logistic', n_jobs=-1, eval_metric='auc')

# Define parameter grid for GridSearch
test_par = {
    'max_depth': [2, 10, 20],
    'learning_rate': [0.1, 0.2, 0.5],
    'n_estimators': [5, 10, 20],
    'min_child_weight': [0.05, 0.15]
}

# Using StratifiedKFold for cross-validation
cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Prepare to store ROC metrics
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
fig = go.Figure()

# Cross-validation loop
for i, (train_idx, test_idx) in enumerate(cv_splitter.split(X_train_scaled, y_train)):
    X_train_cv, X_test_cv = X_train_scaled[train_idx], X_train_scaled[test_idx]
    y_train_cv, y_test_cv = y_train[train_idx], y_train[test_idx]

    # Fit the model
    model = xgb.XGBClassifier(**XGB_gsearch.best_params_, objective='binary:logistic', eval_metric='auc')
    model.fit(X_train_cv, y_train_cv)

    # Predict probabilities
    y_pred_proba = model.predict_proba(X_test_cv)[:, 1]
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_test_cv, y_pred_proba)
    roc_auc = roc_auc_score(y_test_cv, y_pred_proba)
    aucs.append(roc_auc)
    
    # Interpolate the ROC curve
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0

    # Plot the individual ROC curve for each fold
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'Fold {i + 1} (AUC = {roc_auc:.2f})'))

# Compute the mean ROC curve
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = roc_auc_score(y_train_cv, model.predict_proba(X_train_cv)[:, 1])

# Plot the mean ROC curve
fig.add_trace(go.Scatter(x=mean_fpr, y=mean_tpr, mode='lines', name=f'Mean ROC (AUC = {mean_auc:.2f})', line=dict(color='black', dash='dash')))

# Add plot details
fig.update_layout(
    title="Pooled ROC Curve Across Folds",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    showlegend=True
)

# Show the plot
fig.show()

####################################################################
# Feature Importance
####################################################################
# Feature importance
X_feature = X_train.columns  # Assuming X_train is the original feature DataFrame
RF_feat = pd.DataFrame(data=model.feature_importances_, index=X_feature, columns=['importance'])

# Plot feature importance
RF_feat.sort_values('importance').plot(kind='barh', title='XGB Feature Importance')
plt.show()

2