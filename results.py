import os
import re
import collections
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud
from nltk.corpus import stopwords
import pandas as pd
import statsmodels.api as sm
from scipy.stats import ttest_ind, pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def extract_results(text, filename):
    """Extracts relevant voting statistics from the text."""
    accuracy_before = re.search(r'Accuracy before debate: (\d+\.\d+)%', text)
    accuracy_after = re.search(r'Accuracy after debate: (\d+\.\d+)%', text)
    actual_result = re.search(r'Actual Result: (\w+)', text)
    simulated_before = re.search(r'Simulated Result Before Debate: (\w+)', text)
    simulated_after = re.search(r'Simulated Result After Debate: (\w+)', text)

    votes_before = re.findall(r'Initial Votes \(Before Debate\):\n([\s\S]+?)\n\nFinal Votes', text)
    votes_after = re.findall(r'Final Votes \(After Debate\):\n([\s\S]+?)$', text)
    vote_changes = re.search(r'(-?\d+\.\d+)% Yea \+(-?\d+\.\d+)% Nay', text)

    def parse_votes(vote_text):
        votes = {"yea": [], "nay": []}
        if vote_text:
            for line in vote_text[0].split("\n"):
                match = re.match(r'(.+?): (Yea|Nay)', line)
                if match:
                    senator, vote = match.groups()
                    votes[vote.lower()].append(senator)
        return votes

    return {
        "filename": filename.replace(".txt", ""),
        "accuracy_before": float(accuracy_before.group(1)) if accuracy_before else None,
        "accuracy_after": float(accuracy_after.group(1)) if accuracy_after else None,
        "actual_result": actual_result.group(1) if actual_result else None,
        "simulated_before": simulated_before.group(1) if simulated_before else None,
        "simulated_after": simulated_after.group(1) if simulated_after else None,
        "votes_before": parse_votes(votes_before),
        "votes_after": parse_votes(votes_after),
        "yea_change": float(vote_changes.group(1)) if vote_changes else 0.0,
        "nay_change": float(vote_changes.group(2)) if vote_changes else 0.0
    }


def analyze_text(text):
    """Extracts most common phrases excluding stopwords."""
    stop_words = set(stopwords.words('english'))
    words = re.findall(r'\b\w+\b', text.lower())
    filtered_words = [word for word in words if word not in stop_words]
    return collections.Counter(filtered_words).most_common(20)


def generate_visuals(all_results, common_phrases):
    """Generates all required visualizations."""
    filenames = [r["filename"] for r in all_results]
    accuracies_before = [r["accuracy_before"] for r in all_results if r["accuracy_before"] is not None]
    accuracies_after = [r["accuracy_after"] for r in all_results if r["accuracy_after"] is not None]

    avg_before = np.mean(accuracies_before) if accuracies_before else 0
    avg_after = np.mean(accuracies_after) if accuracies_after else 0

    plt.figure(figsize=(6, 4))
    bars = plt.bar(["Before Debate", "After Debate"], [avg_before, avg_after], color=['blue', 'red'])
    plt.ylabel("Average Accuracy (%)")
    plt.title("Average Accuracy Before and After Debate")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, f"{yval:.2f}%", ha='center', va='bottom')
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.plot(filenames, accuracies_before, marker='o', linestyle='-', color='blue', label="Before Debate")
    plt.plot(filenames, accuracies_after, marker='o', linestyle='-', color='red', label="After Debate")
    plt.xlabel("Measure Name")
    plt.ylabel("Accuracy (%)")
    plt.xticks(rotation=90)
    plt.legend()
    plt.title("Accuracy Distribution Before and After Debate")
    plt.show()

    match_before = sum(1 for r in all_results if r.get("simulated_before") == r.get("actual_result"))
    match_after = sum(1 for r in all_results if r.get("simulated_after") == r.get("actual_result"))
    total_bills = len(all_results)
    match_before_pct = (match_before / total_bills) * 100 if total_bills else 0
    match_after_pct = (match_after / total_bills) * 100 if total_bills else 0

    plt.figure(figsize=(6, 4))
    bars = plt.bar(["Before Debate", "After Debate"], [match_before_pct, match_after_pct], color=['blue', 'red'])
    plt.ylabel("Match Percentage (%)")
    plt.title("Percentage of Simulated Results Matching Actual Outcome")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, f"{yval:.2f}%", ha='center', va='bottom')
    plt.show()

    phrases, counts = zip(*common_phrases)
    plt.figure(figsize=(10, 5))
    plt.barh(phrases, counts, color='green')
    plt.xlabel("Count")
    plt.title("Top 20 Phrases Used Across All Bills")
    plt.gca().invert_yaxis()
    plt.show()


basic_simulation_accuracies = {
    "H.J.Res.30": 62.50, "H.J.Res.26": 14.74, "S.316": 80.21, "H.J.Res.7": 27.47, "H.J.Res.27": 59.38,
    "S.870": 97.94, "S.J.Res.11": 58.59, "S.J.Res.9": 33.67, "H.J.Res.39": 62.89, "S.J.Res.24": 52.00,
    "S.J.Res.23": 61.00, "H.R.3746": 29.29, "H.J.Res.45": 46.94, "H.J.Res.44": 59.60, "S.J.Res.42": 54.64,
    "H.R.662": 42.71, "H.R.6363": 87.76, "S.J.Res.43": 60.61, "H.R.7463": 90.00, "S.3853": 69.70,
    "S.J.Res.62": 25.26, "H.R.7888": 71.28, "H.J.Res.98": 51.02, "S.J.Res.61": 60.00, "S.4072": 44.90,
    "S.J.Res.57": 60.00, "H.J.Res.109": 63.27, "S.J.Res.58": 47.37, "H.R.10545": 88.54, "H.R.82": 80.21
}


def plot_comparison_accuracy(all_results):
    """Plots a comparison of accuracy between basic simulation and text file results."""
    filtered_results = [r for r in all_results if r["filename"] in basic_simulation_accuracies]
    filenames = [r["filename"] for r in filtered_results]
    accuracies_textfile = [r["accuracy_after"] for r in filtered_results]
    accuracies_basic = [basic_simulation_accuracies[r["filename"]] for r in filtered_results]

    plt.figure(figsize=(12, 5))
    plt.plot(filenames, accuracies_basic, marker='o', linestyle='-', color='blue', label="Basic Simulation")
    plt.plot(filenames, accuracies_textfile, marker='o', linestyle='-', color='red', label="Advanced Simulation")
    plt.xlabel("Measure Name")
    plt.ylabel("Accuracy (%)")
    plt.xticks(rotation=90)
    plt.legend()
    plt.title("Comparison of Basic vs. Advanced Simulation Accuracy")
    plt.show()


def plot_passage_rates(all_results):
    """Calculates and plots the percentage of bills passed before and after debate."""
    passed_before = sum(1 for r in all_results if r["simulated_before"] == "passed")
    passed_after = sum(1 for r in all_results if r["simulated_after"] == "passed")
    total_bills = len(all_results)

    passed_before_pct = (passed_before / total_bills) * 100 if total_bills else 0
    passed_after_pct = (passed_after / total_bills) * 100 if total_bills else 0

    plt.figure(figsize=(6, 4))
    bars = plt.bar(["Before Debate", "After Debate"], [passed_before_pct, passed_after_pct], color=['blue', 'red'])
    plt.ylabel("Passage Rate (%)")
    plt.title("Percentage of Bills Passed Before and After Debate")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, f"{yval:.2f}%", ha='center', va='bottom')
    plt.show()


def plot_yea_votes(all_results):
    """Plots the total number of 'Yea' votes before and after the debate."""
    total_yea_before = sum(len(r["votes_before"]["yea"]) for r in all_results)
    total_yea_after = sum(len(r["votes_after"]["yea"]) for r in all_results)

    plt.figure(figsize=(6, 4))
    bars = plt.bar(["Before Debate", "After Debate"], [total_yea_before, total_yea_after], color=['blue', 'red'])
    plt.ylabel("Total 'Yea' Votes")
    plt.title("Total 'Yea' Votes Before and After Debate")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, f"{yval}", ha='center', va='bottom')
    plt.show()


def plot_vote_changes(all_results):
    """Plots the average change in Yea and Nay votes."""
    yea_changes = [r["yea_change"] for r in all_results]
    nay_changes = [r["nay_change"] for r in all_results]

    avg_yea_change = np.mean(yea_changes) if yea_changes else 0
    avg_nay_change = np.mean(nay_changes) if nay_changes else 0

    plt.figure(figsize=(6, 4))
    bars = plt.bar(["Yea Votes", "Nay Votes"], [avg_yea_change, avg_nay_change], color=['blue', 'red'])
    plt.ylabel("Average Change (%)")
    plt.title("Average Change in Yea and Nay Votes After Debate")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, f"{yval:.2f}%", ha='center', va='bottom')
    plt.show()


def plot_top_accuracies(all_results):
    """Plots the top 5 highest and lowest accuracy bills before and after the debate."""
    sorted_results = sorted(
        [r for r in all_results if r["accuracy_before"] is not None and r["accuracy_after"] is not None],
        key=lambda x: x["accuracy_before"])
    lowest_5 = sorted_results[:5]
    highest_5 = sorted_results[-5:]

    labels = [r["filename"] for r in lowest_5 + highest_5]
    accuracies_before = [r["accuracy_before"] for r in lowest_5 + highest_5]
    accuracies_after = [r["accuracy_after"] for r in lowest_5 + highest_5]

    x = np.arange(len(labels))
    width = 0.4

    plt.figure(figsize=(12, 6))
    bars1 = plt.bar(x - width / 2, accuracies_before, width, color='red', label='Before Debate')
    bars2 = plt.bar(x + width / 2, accuracies_after, width, color='blue', label='After Debate')

    plt.xlabel("Bill Name")
    plt.ylabel("Accuracy (%)")
    plt.title("Top 5 Highest and Lowest Accuracy Bills Before and After Debate")
    plt.xticks(ticks=x, labels=labels, rotation=45, ha='right')
    plt.legend()

    for bar in bars1 + bars2:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, f"{yval:.2f}%", ha='center', va='bottom', fontsize=8)

    plt.show()


def plot_accuracy_by_year(all_results, bills_df):
    """Plots accuracy before and after debate for votes in 2023 vs. 2024."""
    # Extract accuracy data
    accuracy_df = pd.DataFrame(all_results)[["filename", "accuracy_before", "accuracy_after"]]

    merged_df = accuracy_df.merge(bills_df[["measure_number", "year"]], left_on="filename", right_on="measure_number", how="left")

    # Filter for 2023 and 2024 only
    year_filtered = merged_df[merged_df["year"].isin([2023, 2024])]

    # Compute average accuracy before and after debate for each year
    avg_accuracy = year_filtered.groupby("year")[["accuracy_before", "accuracy_after"]].mean()

    # Plot results
    plt.figure(figsize=(6, 4))
    bars = plt.bar(avg_accuracy.index.astype(str), avg_accuracy["accuracy_before"], width=0.4, label="Before Debate", color='blue', align='center')
    bars2 = plt.bar(avg_accuracy.index.astype(str), avg_accuracy["accuracy_after"], width=0.4, label="After Debate", color='red', align='edge')

    plt.xlabel("Year")
    plt.ylabel("Average Accuracy (%)")
    plt.title("Accuracy Before and After Debate (2023 vs. 2024)")
    plt.legend()

    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.2f}%", ha='center', va='bottom')

    for bar in bars2:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.2f}%", ha='center', va='bottom')

    plt.show()


def main():
    directory = "results"
    all_results = []
    common_phrases = collections.Counter()

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), "r", encoding="utf-8", errors="replace") as file:
                text = file.read()
                results = extract_results(text, filename)
                words = analyze_text(text)
                all_results.append(results)
                common_phrases.update(dict(words))

    # Load the datasets
    bills_df = pd.read_excel("senators_data.xlsx", sheet_name="bills_data")
    accuracy_df = pd.read_excel("accuracy_results.xlsx")
    # Convert vote_date to datetime and extract the year
    bills_df["vote_date"] = pd.to_datetime(bills_df["vote_date"], errors="coerce")
    bills_df["year"] = bills_df["vote_date"].dt.year

    generate_visuals(all_results, common_phrases.most_common(20))
    plot_comparison_accuracy(all_results)
    plot_passage_rates(all_results)
    plot_top_accuracies(all_results)
    plot_vote_changes(all_results)
    plot_accuracy_by_year(all_results, bills_df)

    # REGRESSION ANALYSIS
    # Clean 'type_vote' column
    bills_df["type_vote"] = bills_df["type_vote"].replace(
        {r".*Veto.*": "On Overriding the Veto", r".*Discharge.*": "On Motion to Discharge"}, regex=True
    )
    # Binary encode final_vote_result (1 if passed, 0 if rejected)
    bills_df["final_vote_result"] = bills_df["vote_result"].map({"passed": 1, "rejected": 0})
    bills_df["previous_action_length"] = len(bills_df["previous_action"])
    bills_df["measure_summary_length"] = len(bills_df["measure_summary"])

    # Merge datasets on measure_number and measure_name
    merged_df = accuracy_df.merge(bills_df, on="measure_number", how="left")
    # Convert categorical features to dummy variables
    categorical_features = ["introduced_party", "topic", "type_vote"]
    merged_df = pd.get_dummies(merged_df, columns=categorical_features, drop_first=True)

    # Select relevant features
    # Get updated feature list after dummy encoding
    features = ["year", "final_vote_result", "previous_action_length", "required_majority", "measure_summary_length", "yea", "nay", "not_voting", "num_cosponsors"] + [col for col in merged_df.columns if
                                                                   any(cat in col for cat in categorical_features)]
    target_variables = ["before_debate", "after_debate", "basic_simulation"]

    # Regression Analysis
    for target in target_variables:
        # Select features and add constant for regression
        temp_df = merged_df.drop(columns=[t for t in target_variables if t != target])
        # Drop rows with missing values
        temp_df = temp_df.dropna()
        X = temp_df[features]
        X = X.astype(int)
        X = sm.add_constant(X)

        y = temp_df[target]

        # Run regression
        model = sm.OLS(y, X).fit()
        print(f"\nRegression results for {target}:")
        print(model.summary())

        # Compute correlation matrix
        corr = temp_df[[target] + features].corr()
        # Extract correlations between the target and features
        target_corr = corr[target].drop(target).abs().sort_values(ascending=False)
        # Print the 10 highest correlations with the target
        print(f"\nTop 10 Highest Correlations with {target}:")
        print(target_corr.head(10))


if __name__ == "__main__":
    main()