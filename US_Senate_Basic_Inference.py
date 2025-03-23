import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
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
from sklearn.inspection import permutation_importance

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
print(f"The DataFrame has {df.shape[0]} rows and {df.shape[1]} columns.")

# TEST DIFFERENT MODELS

# Define models to test
models = [
    BernoulliNB(),
    KNeighborsClassifier(),
    LogisticRegression(solver='saga', max_iter=500),
    LinearSVC(),
    SVC(kernel="rbf"),
    DecisionTreeClassifier(),
    ExtraTreeClassifier(),
    ExtraTreesClassifier(),
    AdaBoostClassifier(algorithm='SAMME'),
    RandomForestClassifier(),
    Perceptron(),
    MLPClassifier(solver='adam', hidden_layer_sizes=(100,), max_iter=500)
]


# 1. Preprocessing, getting the binary target 'vote' (Yea or Nay) and performing EDA
def preprocess_data(df):
    # EXTRACTING FEATURES OUT OF 'vote_date'
    # Convert to 'vote_date' to datetime format
    df['vote_date'] = pd.to_datetime(df['vote_date'], format='%m/%d/%Y')
    # Extract useful features
    df['year'] = df['vote_date'].dt.year  # Year of the vote
    df['month'] = df['vote_date'].dt.month  # Month of the vote (seasonal trends)
    # Election season (July to November of an election year)
    df['is_election_season'] = ((df['vote_date'].dt.year == 2024) & (df['vote_date'].dt.month >= 7)).astype(int)
    # Define session boundaries
    session_cutoff = pd.to_datetime('1/3/2024', format='%m/%d/%Y')
    # Create session feature (1 for first session, 2 for second)
    df['session'] = (df['vote_date'] >= session_cutoff).astype(int) + 1

    # PROCESSING 'vote'
    # Drop rows where 'vote' is 'Not Voting' and filter relevant columns
    df = df[df['vote'] != 'Not Voting']
    # Convert 'vote' into a binary target (Yea = 1, Nay = 0)
    df.loc[:, 'vote'] = df['vote'].map({'Yea': 1, 'Nay': 0})

    # Create final_vote column: 1 if it's a final vote, else 0
    df['final_vote'] = (df['type_vote'] == "On Passage of the Bill").astype(int)

    # List of religions to group as "Protestant"
    protestant_religions = [
        'Church of Christ', 'Church of God', 'Congregationalist', 'Episcopalian',
        'Evangelical', 'Evangelical Protestant', 'Lutheran', 'Methodist',
        'Nondenominational', 'Pentecostal', 'Presbyterian', 'Protestant',
        'Southern Baptist', 'United Church of Christ', 'United Methodist'
    ]
    # Apply transformation
    df['grouped_religion'] = np.where(df['religion'].isin(protestant_religions), 'Protestant', df['religion'])

    # Creating ratios for Voting Behavior
    df['party_loyalty'] = df['AllVoteW'] / df['TotalAll']
    df['party_defection'] = df['AllVoteO'] / df['TotalAll']
    df['key_vote_loyalty'] = df['KeyVoteW'] / df['TotalKey']
    df['key_vote_defection'] = df['KeyVoteO'] / df['TotalKey']

    # Select relevant columns from senator and bill data
    df = df.drop(
        columns=['vote_date', 'measure_number', 'vote_result', 'previous_action',
                 'type_vote', 'measure_title', 'measure_summary', 'bill_text',
                 'yea', 'nay', 'not_voting', 'sponsor',
                 'first_name', 'last_name', 'date_of_birth', 'human_bio', 'religion', 'start', 'end',
                 'served_house', 'served_senate', 'education', 'state',
                 'TotalAll', 'AllVoteW', 'AllVoteO',
                 'TotalKey', 'KeyVoteW', 'KeyVoteO',
                 'Party Unity Opposition (PUO)',
                 'Presidential Opposition (PSO)',
                 'key_vote_loyalty', 'Presidential Support (PSS)', 'year'])  # Drop columns that shouldn't be features

    # Drop any rows that have missing values in the feature or target columns
    df = df.dropna()

    # Label Encoding for 'party', 'introduced_party', 'education_category' and 'state_direction'
    # Initialize LabelEncoder
    le = LabelEncoder()
    # Apply label encoding
    for col in ['party', 'introduced_party', 'state_direction']:
        df[col] = le.fit_transform(df[col])
    education_order = ["associates", "undergraduate", "Masters", "postgraduate"]
    df['education_category'] = df['education_category'].apply(lambda x: education_order.index(x))

    # One-Hot Encoding for other categorical variables
    df = pd.get_dummies(df, columns=['grouped_religion', 'race', 'topic'], drop_first=True)

    # Drop target variable from features
    features = df.drop(columns='vote')

    print("-----------FEATURES ARE-------------")
    print(features.columns)

    return df, features, df['vote'].astype(int)


# Function to find correlations
def perform_eda(df):
    df, features, target = preprocess_data(df)
    # 1. Basic info
    print("\n🔹 Dataset Info:")
    print(df.info())

    # 2. Summary statistics
    print("\n🔹 Summary Statistics:")
    print(df.describe())

    # 3. Check for missing values
    missing_values = df.isnull().sum()
    print("\n🔹 Missing Values:")
    print(missing_values[missing_values > 0])

    # 4. Correlation matrix
    # Selecting relevant numerical features
    num_features = [
        'required_majority', 'num_cosponsors', 'party_loyalty',
        'Party Unity Support (PUS)', 'Voting Participation (VP)', 'bipartisan_index',
        'dw_nominate', 'years_senate', 'years_house', 'age', 'state_pvi', 'month', 'is_election_season', 'session',
        'final_vote'
    ]

    # Compute correlation
    corr_matrix = df[num_features + ['vote']].corr()

    # Mask for the upper triangle (hides redundant correlations)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Plot heatmap with smaller font
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt=".2f",
                linewidths=0.5, annot_kws={"size": 8})  # Smaller font size

    plt.title("Cleaned Correlation Matrix (Lower Triangle Only)", fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()

    # Pairplots
    # Select key features for pair plots
    features_to_plot = ['bipartisan_index', 'dw_nominate', 'state_pvi', 'party_loyalty', 'vote']
    # Generate pair plot
    sns.pairplot(df[features_to_plot], hue="vote", palette="Set1", diag_kind="kde")
    # Show plot
    plt.show()


# 2. Split the data into training and testing sets
def split_data(df):
    df, features, target = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Use same scaler from training for test data

    return X_train_scaled, X_test_scaled, y_train, y_test


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

    # Compute permutation importance
    perm_importance = permutation_importance(model, test_x, test_y, scoring='accuracy', n_repeats=10, random_state=42)

    # Get feature importance scores
    feature_importance_df = pd.DataFrame({
        'Feature': all_data.drop(columns='vote').columns,
        'Importance': perm_importance.importances_mean
    }).sort_values(by='Importance', ascending=False)

    print(f'Feature Importance for {model.__class__.__name__}:')
    print(feature_importance_df.head(10))  # Print top 10 features

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10), palette='viridis')
    plt.title(f'Feature Importance ({model.__class__.__name__})')
    plt.show()

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


all_data, features, target = preprocess_data(df)

# 5. Run training and testing for each model
def run_test():
    X_train, X_test, y_train, y_test = split_data(df)
    for model in models:
        accuracy = train_model(X_train, y_train, model)
        print(
            f'Model: {model.__class__.__name__}, Cross-validation Accuracy: {accuracy.mean():.2f} ± {accuracy.std():.2f}')


def run_test2():
    X_train, X_test, y_train, y_test = split_data(df)
    for model in models:
        accuracy = test_model(model, X_train, y_train, X_test, y_test)
        print(f'Model: {model.__class__.__name__}, Test Accuracy: {accuracy:.2f}')


# Run the EDA on the dataset
# perform_eda(df)

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

# Get training features and target
# X_train, X_test, y_train, y_test = split_data(df)

# Perform Grid Search
# perform_grid_search_cv(param_grid, RandomForestClassifier(), X_train, y_train)
