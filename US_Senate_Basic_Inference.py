import numpy
import pandas as pd
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, KFold
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_curve, auc

data_file = 'senators_data.xlsx'

# Read the senators data
senators_df = pd.read_excel(data_file, sheet_name='senators_data')
# Display the first few rows of the dataframe
print('--------SENATORS DATA--------')
print(senators_df.head())

# Read the votes data
votes_df = pd.read_excel(data_file, sheet_name='bills_data')
# Display the first few rows of the dataframe
print('--------VOTES DATA--------')
print(senators_df.head())

# Create a dictionary to store DataFrames corresponding to each measure_number
measure_dataframes = {}
sheet_names = pd.ExcelFile(data_file).sheet_names

# Iterate over each measure_number in votes_df
for measure_number in votes_df['measure_number']:
    # Check if the sheet exists in the workbook
    if measure_number in sheet_names:
        # Read the sheet into a DataFrame
        measure_df = pd.read_excel(data_file, sheet_name=measure_number)
        # Drop party and state columns
        measure_df = measure_df.drop(columns=['party', 'state'])

        # Store the DataFrame in the dictionary using the measure_number as the key
        measure_dataframes[measure_number] = measure_df
    else:
        print(f"Sheet for measure_number {measure_number} does not exist in the workbook.")

# COMBINE INDIVIDUAL VOTES ON EACH BILL WITH OVERALL DETAILS OF BILLS
# Create a list to store the combined DataFrames
combined_dataframes = []

# Iterate over each DataFrame in the measure_dataframes dictionary
for measure_number, measure_df in measure_dataframes.items():
    # Add the measure_number as a new column in the measure_df
    measure_df['measure_number'] = measure_number

    # Find the corresponding bill details for the current measure_number
    bill_details = votes_df[votes_df['measure_number'] == measure_number]

    # Merge the measure_df with the bill details on the measure_number
    merged_df = pd.merge(measure_df, bill_details, on='measure_number', how='left')

    # Append the merged DataFrame to the list
    combined_dataframes.append(merged_df)

# Concatenate all the combined DataFrames into one large DataFrame
combined_bill_df = pd.concat(combined_dataframes, ignore_index=True)

# COMBINE BILL DETAILS WITH DETAILS OF SENATORS
# Merge with senators_data based on last_name
df = pd.merge(combined_bill_df, senators_df, on='last_name', how='left')

# Display the first few rows of the final DataFrame
print('--------COMBINED DATA--------')
print(df.head())
# Print all the column names in the final DataFrame
print(df.columns)


# TEST DIFFERENT MODELS

# Define models to test
models = [
    BernoulliNB(),
    KNeighborsClassifier(),
    LogisticRegression(),
    LinearSVC(),
    SVC(kernel="rbf"),
    DecisionTreeClassifier(),
    ExtraTreeClassifier(),
    ExtraTreesClassifier(),
    AdaBoostClassifier(),
    RandomForestClassifier(),
    Perceptron(),
    MLPClassifier()
]


# 1. Preprocessing, getting the binary target 'vote' (Yea or Nay) and performing EDA
def preprocess_data(df):
    # Drop rows where 'vote' is 'Not Voting' and filter relevant columns
    df = df[df['vote'] != 'Not Voting']

    # Convert 'vote' into a binary target (Yea = 1, Nay = 0)
    df.loc[:, 'vote'] = df['vote'].map({'Yea': 1, 'Nay': 0})

    # Select relevant columns from senator and bill data
    df = df.drop(
        columns=['measure_number', 'first_name', 'last_name', 'date_of_birth',
                 'education', 'human_bio', 'start', 'end', 'served_house', 'served_senate',
                 'vote_date', 'measure_number', 'vote_result', 'measure_title', 'measure_summary',
                 'yea', 'nay', 'not_voting', 'introduced_by', 'Presidential Support (PSS)', 'Presidential Opposition (PSO)', 'Party Unity Support (PUS)', 'Party Unity Opposition (PUO)', 'Voting Participation (VP)', 'last_election',
                 'TotalAll', 'AllVoteW', 'AllVoteO', 'TotalKey', 'KeyVoteW', 'KeyVoteO'])  # Drop columns that shouldn't be features

    # Drop any rows that have missing values in the feature or target columns
    df = df.dropna()

    # Drop target variable from features
    features = df.drop(columns='vote')

    # One-Hot Encoding for categorical variables
    features = pd.get_dummies(features, columns=['party', 'state', 'religion', 'race', 'introduced_party', 'topic', 'education_category', 'state_direction'], drop_first=True)

    return features, df['vote']


# Function to perform exploratory data analysis (EDA)
def perform_eda(df, x_list, y_list):
    print(df.info())
    print(df.head(5))
    print(df.describe())
    sns.pairplot(df, x_vars=x_list, y_vars=y_list, hue='party', palette='Set1')
    plt.show()


# 2. Split the data into training and testing sets
def split_data(df):
    features, target = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


# 3. Perform model training and evaluation
def train_model(x, y, model, test_size=10):
    pipe = Pipeline(
        [
            ("classifier", model),
        ],
    )
    kf = KFold(n_splits=test_size, shuffle=True, random_state=0)
    cross_val_results = cross_val_score(pipe, x, y, cv=kf)
    return cross_val_results


# 4. Testing model and printing performance metrics
def plot_roc_curve(test_y, probabilities, model_name="Model"):
    """
    Plots the ROC curve and calculates the AUC for binary classification.
    """
    fpr, tpr, thresholds = roc_curve(test_y, probabilities)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {model_name}')
    plt.legend(loc="lower right")
    plt.show()

    print(f'ROC AUC: {roc_auc:.2f}')
    return roc_auc


def test_model(model, train_x, train_y, test_x, test_y):
    model.fit(train_x, train_y)
    predictions = model.predict(test_x)
    accuracy = accuracy_score(test_y, predictions)

    # F1 Score
    f1 = f1_score(test_y, predictions, average='weighted')
    print(f'F1 Score: {f1}')

    # Calculate probabilities for the positive class
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(test_x)[:, 1]
    else:
        probabilities = model.decision_function(test_x)
        probabilities = (probabilities - probabilities.min()) / (probabilities.max() - probabilities.min())  # Normalize

    # Plot ROC Curve
    plot_roc_curve(test_y, probabilities, model_name=model.__class__.__name__)

    return accuracy


# 5. Run training and testing for each model
def run_test():
    X_train, X_test, y_train, y_test = split_data(df)
    for model in models:
        accuracy = train_model(X_train, y_train, model)
        print(f'Model: {model.__class__.__name__}, Cross-validation Accuracy: {accuracy.mean():.2f} ± {accuracy.std():.2f}')


def run_test2():
    X_train, X_test, y_train, y_test = split_data(df)
    for model in models:
        accuracy = test_model(model, X_train, y_train, X_test, y_test)
        print(f'Model: {model.__class__.__name__}, Test Accuracy: {accuracy:.2f}')


# Run the EDA on the dataset
y_vars = ['age', 'dw_nominate', 'bipartisan_index']
x_vars = ['vote']
perform_eda(df, x_vars, y_vars)

# Running both test functions
# run_test()
# run_test2()


def perform_grid_search_cv(params, model, x, y):
    grid_search = GridSearchCV(model, params, cv=10, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search.fit(x, y)
    print(f'Best params: {grid_search.best_params_}')
    print(f'Best score: {grid_search.best_score_}')


param_grid = {
    'n_estimators': [1, 10, 50, 100],
    'max_features': ['sqrt', 'log2'],
    'ccp_alpha': [0.1, .01, .001, 0.0001],
    'max_depth': [10, 15, 20, 50, 100, None],
    'criterion': ['gini', 'entropy', 'log_loss']
}

# perform_grid_search_cv(param_grid, ExtraTreesClassifier(), training_features, targets['skill_level2'])
