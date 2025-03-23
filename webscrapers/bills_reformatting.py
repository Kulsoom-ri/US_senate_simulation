import pandas as pd
import re

# script to parse roll call votes for bills scraped from congress.gov
# and arrange them into columns

# File path for input and output
data_file = 'senators_data.xlsx'
output_file = 'formatted_votes_data.xlsx'

# Read the votes data
votes_df = pd.read_excel(data_file, sheet_name='bills_data')

# Get sheet names
sheet_names = pd.ExcelFile(data_file).sheet_names

# Name corrections
name_corrections = {
    "Casey": "Casey Jr.",
    "King": "King Jr.",
    "Manchin": "Manchin III"
}


def parse_vote_entry(entry):
    """Extract last name, party, state, and vote from the entry."""
    match = re.match(r"(.*) \((.)-(..)\),\s*(.*)", entry)
    if match:
        last_name, party, state, vote = match.groups()
        # Apply name correction if needed
        last_name = name_corrections.get(last_name, last_name)
        return [last_name, party, state, vote]
    return [None, None, None, None]


# Dictionary to store reformatted DataFrames
reformatted_sheets = {}

for measure_number in votes_df['measure_number']:
    # Skip "S.316 - Passage" sheet
    # if measure_number == "S.316 - Passage":
    #    print(f"Skipping sheet: {measure_number}")
    #    continue

    if measure_number in sheet_names:
        # Read the sheet
        measure_df = pd.read_excel(data_file, sheet_name=measure_number, header=None)

        # Process data
        parsed_data = [parse_vote_entry(entry) for entry in measure_df.iloc[:, 0]]

        # Create new DataFrame
        formatted_df = pd.DataFrame(parsed_data, columns=['last_name', 'party', 'state', 'vote'])

        # Store in dictionary
        reformatted_sheets[measure_number] = formatted_df

# Write to a new Excel workbook
with pd.ExcelWriter(output_file) as writer:
    for sheet_name, df in reformatted_sheets.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print("Reformatted data has been saved to", output_file)