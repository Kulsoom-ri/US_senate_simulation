from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import time
import pandas as pd
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Load the Excel file containing names
excel_path = "senators_data.xlsx"
df = pd.read_excel(excel_path, engine="openpyxl")
df = df.iloc[84:]

# Path to the directory containing your Chrome profile
profile_path = "C:\\Users\\super\\AppData\\Local\\Google\\Chrome\\User Data"

# Specify the profile name you want to use, for example 'Profile 1' or 'Default'
profile_name = "Profile 1"

options = Options()
options.add_argument(f"user-data-dir={profile_path}")  # Path to the User Data folder
options.add_argument(f"profile-directory={profile_name}")  # Profile directory (e.g., 'Profile 1')

for index, row in df.iterrows():
    # Initialize the webdriver
    driver = webdriver.Chrome(executable_path='chromedriver.exe', options=options)
    first_name = row["first_name"]
    last_name = row["last_name"]
    full_name = f"{last_name}, {first_name}"
    try:
        # Open the webpage
        driver.get("https://library-cqpress-com.proxy.lib.duke.edu/congress/votingalignment.php")
        time.sleep(13)  # Allow time for login/authentication if needed
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        # Click the dropdown
        dropdown = driver.find_element(By.ID, "member3_chosen")
        dropdown.click()
        time.sleep(1)

        # Try to select the member
        option = driver.find_element(By.XPATH, f"//li[contains(text(), '{full_name}')]")
        option.click()
        print(f"Selected: {full_name}")

        # Click the 'Find Party Unity' button
        find_party_unity_button = driver.find_element(By.XPATH, "//button[contains(text(), 'FIND PARTY UNITY')]")
        find_party_unity_button.click()
        print("Clicked on 'FIND PARTY UNITY' button.")
        time.sleep(2)

        # Click the 'Export Data' button
        export_data_link = driver.find_element(By.LINK_TEXT, "Export data")
        export_data_link.click()
        print("Clicked on 'Export data' link.")
        time.sleep(2)

        # Check the "I agree" checkbox
        checkbox = driver.find_element(By.ID, "license")
        checkbox.click()

        # Click the 'EXPORT DATA' button
        export_button = driver.find_element(By.XPATH, "//button[contains(text(), 'EXPORT DATA')]")
        export_button.click()
        print(f"Export process started for {full_name}.")
        time.sleep(2)  # Wait for download to complete before processing the next person

    except NoSuchElementException:
        print(f"Skipping: {full_name} not found in the list.")
        continue  # Skip to the next person in the list

    finally:
        driver.quit()
        time.sleep(2)

print("Process completed.")