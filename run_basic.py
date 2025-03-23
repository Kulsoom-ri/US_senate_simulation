from groq import Groq
import pandas as pd
import os
import random
import re
import openpyxl

# Set random seed for reproducibility
random.seed(42)

# Load data
data_file = 'senators_data.xlsx'
senators_df = pd.read_excel(data_file, sheet_name='senators_data')
votes_df = pd.read_excel(data_file, sheet_name='bills_data')

# Display first few rows
print('--------SENATORS DATA--------')
print(senators_df.head())
print('--------VOTES DATA--------')
print(votes_df.head())

# Filtering to only include bills that involve passage
votes_df = votes_df[votes_df['type_vote'] == 'On Passage of the Bill']

# Load vote data per bill
measure_dataframes = {}
sheet_names = pd.ExcelFile(data_file).sheet_names

for measure_number in votes_df['measure_number']:
    if str(measure_number) in sheet_names:
        measure_df = pd.read_excel(data_file, sheet_name=str(measure_number))
        measure_df = measure_df.drop(columns=['party', 'state'])  # Drop irrelevant columns
        measure_dataframes[measure_number] = measure_df
    else:
        print(f"Sheet for measure_number {measure_number} not found.")

# Extract full names
senators_df['full_name'] = senators_df['first_name'] + " " + senators_df['last_name']
senator_names = senators_df['full_name'].tolist()

# Initialize API client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Track accuracy
global_correct = 0
global_total = 0
bill_outcome_matches = 0  # Track how often the model predicts the correct bill outcome

# Loop through senators and bills
for measure_number, measure_df in measure_dataframes.items():
    # Retrieve the bill details
    bill_row = votes_df[votes_df['measure_number'] == measure_number]
    measure_summary = bill_row['measure_summary'].values[0]
    required_majority = bill_row['required_majority'].values[0]
    actual_vote_result = bill_row['vote_result'].values[0]  # 'passed' or 'rejected'

    print(f"\nProcessing bill {measure_number}...")

    # Track accuracy for this bill
    bill_correct = 0
    bill_total = 0
    simulated_yea_count = 0  # Count of Yea votes in the simulation

    # Filter out senators who did not vote
    voted_senators = measure_df[measure_df['vote'] != 'Not Voting']

    # Loop through senators who voted
    for _, senator in voted_senators.iterrows():
        last_name = senator['last_name']
        actual_vote = senator['vote']

        # Find the full name of the senator
        matched_senator = senators_df[senators_df['last_name'] == last_name]
        if matched_senator.empty:
            print('No match found for: '+last_name)
            continue  # Skip if no matching senator found
        full_name = matched_senator['full_name'].iloc[0]  # Extract full name

        # Query the Groq API for the simulated vote
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": f"You are {full_name}, a US senator."},
                {"role": "user", "content": f"As senator {full_name}, how would you vote on the passage of this bill?\n\n"
                                            f"**Bill Summary:** {measure_summary}\n\n"
                                            f"Only reply with 'Yea' or 'Nay'."}
            ],
            model="llama-3.1-8b-instant",
            seed=42,
            stream=False,
        ).choices[0].message.content.strip().lower()

        if re.search(r'\byea\b', response, re.IGNORECASE):
            response = "yea"
        elif re.search(r'\bnay\b', response, re.IGNORECASE):
            response = "nay"
        else:
            print(f"Unexpected response: {response}")

        # Update accuracy metrics
        bill_total += 1
        global_total += 1
        if response == actual_vote.lower():
            bill_correct += 1
            global_correct += 1

        # Count simulated Yea votes
        if response.lower() == "yea":
            simulated_yea_count += 1

        print(f"{full_name} | Predicted: {response} | Actual: {actual_vote}")

    # Calculate and print accuracy for this bill
    bill_accuracy = (bill_correct / bill_total * 100) if bill_total > 0 else 0
    print(f"\nAccuracy for Bill {measure_number}: {bill_accuracy:.2f}%")

    # Determine if the bill passes based on simulated votes
    total_votes = len(voted_senators)
    required_votes = total_votes * (required_majority / 100)  # Convert percentage to actual vote count
    simulated_vote_result = "passed" if simulated_yea_count >= required_votes else "rejected"

    # Compare with actual result
    bill_result_match = simulated_vote_result == actual_vote_result
    if bill_result_match:
        bill_outcome_matches += 1

    print(
        f"Simulated Outcome: {simulated_vote_result} | Actual Outcome: {actual_vote_result} | Match: {bill_result_match}")

# Print overall accuracy
if global_total > 0:
    global_accuracy = (global_correct / global_total) * 100
    print(f"\nOverall Model Accuracy: {global_accuracy:.2f}%")
else:
    print("\nNo predictions made.")

# Print bill outcome accuracy
total_bills = len(measure_dataframes)
if total_bills > 0:
    outcome_accuracy = (bill_outcome_matches / total_bills) * 100
    print(f"\nBill Outcome Prediction Accuracy: {outcome_accuracy:.2f}%")