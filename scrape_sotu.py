from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

import pickle

sotu_raw_transcripts = []

dates = list(range(1793, 1797)) # Dates 1793-2020
dates_str = [str(d) for d in dates]
dates_str

for date in dates_str:
    print(f'Getting transcript for {date}')
    try:
        # Opening URL
        base_url = 'https://www.presidency.ucsb.edu/documents/presidential-documents-archive-guidebook/annual-messages-congress-the-state-the-union'
        driver = webdriver.Chrome('./chromedriver')
        driver.get(base_url)
        driver.implicitly_wait(10) # seconds
        element = driver.find_element_by_link_text(date)
        webdriver.ActionChains(driver).move_to_element(element).click(element).perform()
        current_url = driver.current_url # url containing transcript of SOTU addresses
        #print(current_url)
        r = requests.get(current_url)
        #print(r.content)
        soup = BeautifulSoup(r.content, 'html5lib') 
        #print(soup.prettify())
        # Using BeautifulSoup to extract transcript
        text = soup.find('div', class_='field-docs-content').text
        text = soup.find('div', class_='node-documents').text
        sotu_raw_transcripts.append(text)
        driver.close()
    except:
        print(f'Could not get record for {date}')
#print(sotu_transcripts)
num_records = len(sotu_raw_transcripts)
print(f'Successfully scrapped {num_records} records')
driver.quit()

# Pickling raw transcripts
with open('./pickled_files/sotu_raw_transcripts.pkl', 'wb') as f:
    pickle.dump(sotu_raw_transcripts, f)