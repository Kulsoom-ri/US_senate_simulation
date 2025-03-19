from network import Network
import csv

csv_file = 'senators_data.csv'  # CSV that stores senator data
# Open the CSV file and read it into a list of dictionaries
with open(csv_file, newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    senators = [row for row in reader]

senators = senators[0:2]

# Create a list of full names and a list of identities
names = [f"{senator['first_name']} {senator['last_name']}" for senator in senators]
identities = [
    {
        'name': f"{senator['first_name']} {senator['last_name']}",
        'party': senator['party'],
        'state': senator['state'],
        'years_senate': senator['years_senate'],
        'dw_nominate': senator['dw_nominate'],
        'bipartisan_index': senator['bipartisan_index'],
        'policy_bio': senator['policy_bio']
    }
    for senator in senators
]


# Setting up the network of the senators
agent_network = Network(num_agents=2, max_context_size=100, names=names, identities=identities)

# Run group chat for debates and then predict voting behavior
prompt = "Discuss the bill on healthcare reform"
question = "Do you support this bill?"
agent_network.group_chat(prompt, "round_robin", max_rounds=1)
agent_network.predict(prompt, question)