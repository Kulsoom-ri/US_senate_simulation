from network import Network
import pandas as pd
import json
import sys

# for printing both to terminal and writing to file
class Tee:
    def __init__(self, filename, mode="w"):
        self.terminal = sys.stdout  # Save original stdout
        self.log = open(filename, mode)  # Open file for writing

    def write(self, message):
        self.terminal.write(message)  # Print to terminal
        self.log.write(message)  # Write to file

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


# Load data
data_file = 'senators_data.xlsx'
senators_df = pd.read_excel(data_file, sheet_name='senators_data')
votes_df = pd.read_excel(data_file, sheet_name='bills_data')
votes_df = votes_df.iloc[38:45]
# Load vote data per bill
measure_dataframes = {}
sheet_names = pd.ExcelFile(data_file).sheet_names

# Set the parameters
VOTE_CUTOFF_DATE = pd.to_datetime('2024-01-03')
max_context_size = 15000

for measure_number in votes_df['measure_number']:
    if str(measure_number) in sheet_names:
        measure_df = pd.read_excel(data_file, sheet_name=str(measure_number))

        # Look up vote_date using measure_number
        vote_date = votes_df.loc[votes_df['measure_number'] == measure_number, 'vote_date']
        if not vote_date.empty:
            vote_date = pd.to_datetime(vote_date.iloc[0])  # Extract and convert
        else:
            print(f"No vote date found for measure number {measure_number}")
            continue

        # Keep only votes before the cutoff date and only 'yea' or 'nay' votes
        measure_df = measure_df[
            (measure_df['vote'].str.lower().isin(['yea', 'nay'])) &
            (vote_date >= VOTE_CUTOFF_DATE)
            ].drop(columns=['party', 'state'], errors='ignore')  # Drop irrelevant columns

        measure_dataframes[measure_number] = measure_df
    else:
        print(f"Sheet for measure_number {measure_number} not found.")

# Loop through each bill and create a network for each
for measure_number, measure_df in measure_dataframes.items():
    output_file = f"results/{measure_number}.txt"

    # Close the previous file and reset stdout before opening a new file
    if isinstance(sys.stdout, Tee):
        sys.stdout.close()  # Close the current file
        sys.stdout = sys.__stdout__  # Restore stdout to terminal

    sys.stdout = Tee(output_file)

    print(f"\n----- Processing Bill {measure_number} -----")

    # Get a mutable set of valid voters (tuples of first and last name)
    valid_voters = set(zip(measure_df['first_name'], measure_df['last_name']))

    # Filter senators who voted on the bill, ensuring each is matched only once
    filtered_senators = [senator for _, senator in senators_df.iterrows() if
                         (senator['first_name'], senator['last_name']) in valid_voters]

    # Generate senator identities, including their actual votes
    identities = []
    for senator in filtered_senators:
        # Find the corresponding vote from measure_df
        vote_row = measure_df[(measure_df['first_name'] == senator['first_name']) &
                              (measure_df['last_name'] == senator['last_name'])]
        vote = vote_row['vote'].values[0] if not vote_row.empty else None  # Get the vote or None if not found

        identities.append({
            'name': f"{senator['first_name']} {senator['last_name']}",
            'age': senator['age'],
            'religion': senator['religion'],
            'education': senator['education'],
            'party': senator['party'],
            'state': senator['state'],
            'state_pvi': f"R+{abs(senator['state_pvi'])}" if senator['state_pvi'] < 0 else f"D+{senator['state_pvi']}",
            'years_senate': senator['years_senate'],
            'years_house': senator['years_house'],
            'last_election': senator['last_election'],
            'party_loyalty': senator['AllVoteW'] / senator['TotalAll'] if senator['TotalAll'] else None,
            'party_unity': senator['Party Unity Support (PUS)'],
            'presidential_support': senator['Presidential Support (PSS)'],
            'voting_participation': senator['Voting Participation (VP)'],
            'dw_nominate': senator['dw_nominate'],
            'bipartisan_index': senator['bipartisan_index'],
            'bio': senator['human_bio'],
            'vote': vote  # Include the actual vote here
        })

    # Skip if no valid senators remain for this bill
    if not identities:
        print(f"Skipping Bill {measure_number} - No eligible senators")
        continue

    # Pretty print identities
    # print(f"Senators in Network for Bill {measure_number}:")
    # print(json.dumps(identities, indent=4, ensure_ascii=False))

    # Setting up the network of the filtered senators
    agent_network = Network(
        num_agents=len(identities),
        max_context_size=max_context_size,
        names=[s['name'] for s in identities],
        identities=identities
    )

    vote_date = votes_df.loc[votes_df['measure_number'] == measure_number, 'vote_date'].values[0]
    measure_summary = votes_df.loc[votes_df['measure_number'] == measure_number, 'measure_summary'].values[0]
    type_vote = votes_df.loc[votes_df['measure_number'] == measure_number, 'type_vote'].values[0]
    sponsor = votes_df.loc[votes_df['measure_number'] == measure_number, 'sponsor'].values[0]
    introduced_party = votes_df.loc[votes_df['measure_number'] == measure_number, 'introduced_party'].values[0]
    num_cosponsors = votes_df.loc[votes_df['measure_number'] == measure_number, 'num_cosponsors'].values[0]
    previous_action = votes_df.loc[votes_df['measure_number'] == measure_number, 'previous_action'].values[0]
    required_majority = votes_df.loc[votes_df['measure_number'] == measure_number, 'required_majority'].values[0]
    vote_result = votes_df.loc[votes_df['measure_number'] == measure_number, 'vote_result'].values[0]

    # Define the debate topic and question
    prompt = (
        f"The date is: {vote_date}\n"
        f"There is a floor vote happening: {type_vote}\n"
        f"The bill under consideration is: {measure_summary}\n"
        f"The bill was introduced by {sponsor}, a {introduced_party} and has {num_cosponsors} cosponsors.\n"
        f"This action has already happened on this bill: {previous_action}"
    )
    question = "Based on the debate, do you support or oppose this vote?"

    # Run the group discussion and predict votes
    agent_network.group_chat(prompt, "round_robin", max_rounds=2)

    majority_threshold = required_majority/100 * len(identities)
    final_pre_yea_count, final_post_yea_count = agent_network.predict(prompt, question, measure_df)
    pre_vote_result = "passed" if final_pre_yea_count >= majority_threshold else "rejected"
    post_vote_result = "passed" if final_post_yea_count >= majority_threshold else "rejected"

    # Print comparison results
    print(f"\n--- Final Outcome Results Comparison for {measure_number}---")
    print(f"Required Majority: {required_majority}%")
    print(f"Simulated Result Before Debate: {pre_vote_result} (Yea: {final_pre_yea_count})")
    print(f"Simulated Result After Debate: {post_vote_result} (Yea: {final_post_yea_count})")
    print(f"Actual Result: {vote_result}")

sys.stdout = sys.__stdout__
