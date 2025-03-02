from selenium import webdriver
import time
from bs4 import BeautifulSoup
import csv

driver = webdriver.Chrome()
url = "https://www.congress.gov/members?q=%7B%22congress%22%3A118%2C%22chamber%22%3A%22Senate%22%7D"
driver.get(url)

# wait for page to load
time.sleep(7)

# Get the page source
page_source = driver.page_source

# Close the browser window
driver.quit()

# Parse the page using BeautifulSoup
soup = BeautifulSoup(page_source, 'html.parser')

# Find all Senator list items
senators = soup.select('li.expanded')

# Create an empty list to store senator data
senator_data = []

# Loop through each senator and extract the necessary information
for senator in senators:
    # Extract the name from the alt attribute of the <img> tag
    img_tag = senator.find('img')
    name = img_tag['alt'] if img_tag else ""

    # Split the name into parts using commas, then handle the case with 3 parts
    first_name, last_name = "", ""
    if name:
        name_parts = name.split(', ')

        # Handle the case where there are 3 parts (e.g., "Casey, Robert P., Jr.")
        if len(name_parts) == 3:
            first_name = name_parts[1].strip()  # The middle part is the first name
            last_name = name_parts[0].strip() + " " + name_parts[
                2].strip()  # Merge the first and last parts for last name
        else:
            first_name = name_parts[1].strip() if len(name_parts) > 1 else ""
            last_name = name_parts[0].strip()

    # Extract state from the <span> after "State:" label
    state_tag = senator.find('strong', string="State:")
    state = ""
    if state_tag:
        state_tag = state_tag.find_next('span')
        state = state_tag.text.strip() if state_tag else ""

    # Extract party from the <span> after "Party:" label
    party_tag = senator.find('strong', string='Party:')
    party = ""
    if party_tag:
        party_tag = party_tag.find_next('span')
        party = party_tag.text.strip() if party_tag else ""

    # Extract the years served (Senate and House) from the <ul class="member-served">
    served_info = senator.find('ul', class_='member-served')
    years_senate = ""
    years_house = ""

    if served_info:
        for li in served_info.find_all('li'):
            if "Senate" in li.text:
                years_senate = li.text.split(":")[1].strip()
            elif "House" in li.text:
                years_house = li.text.split(":")[1].strip()

    # Store the senator's data in the dictionary
    senator_info = {
        "first_name": first_name,
        "last_name": last_name,
        "state": state,
        "party": party,
        "years_house": years_house,
        "years_senate": years_senate
    }

    senator_data.append(senator_info)

# Print the senator data
for senator in senator_data:
    print(senator)

# Save the senator data to a CSV file
csv_file = "senators_data.csv"
with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
    fieldnames = ["first_name", "last_name", "state", "party", "years_house", "years_senate"]
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    # Write the header
    writer.writeheader()

    # Write the senator data
    for senator in senator_data:
        writer.writerow(senator)

print(f"Data saved to {csv_file}")