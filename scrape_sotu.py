from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

import pickle

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

sotu_raw_transcripts = []
presidents = []
sotu_dates = []

dates = list(range(1793, 2021)) # Dates 1793-2020
dates_str = [str(d) for d in dates]
dates_str

for date in dates_str:
    print(f'Getting transcript for {date}')
    try:
        # Opening URL
        base_url = 'https://www.presidency.ucsb.edu/documents/presidential-documents-archive-guidebook/annual-messages-congress-the-state-the-union'
        driver = webdriver.Chrome()
        driver.get(base_url)
        driver.implicitly_wait(10) # seconds
        element = driver.find_element_by_link_text(date)
        webdriver.ActionChains(driver).move_to_element(element).click(element).perform()
        current_url = driver.current_url # url containing transcript of SOTU addresses
        r = requests.get(current_url, headers=headers)
        print(r)
        # print(r.content)
        soup = BeautifulSoup(r.content, 'html.parser')
        #print(soup.prettify())
        # Using BeautifulSoup to extract transcript
        text = soup.find('div', class_='field-docs-content').get_text()
        #text = soup.find('div', class_='node-documents').get_text()
        pres_name = soup.find('h3', class_='diet-title').get_text()
        date = soup.find('span', class_='date-display-single').get_text()
        sotu_raw_transcripts.append(text)
        presidents.append(pres_name)
        sotu_dates.append(date)
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

# Pickling list of presidents
with open('./pickled_files/presidents.pkl', 'wb') as f:
    pickle.dump(presidents, f)

# Opening picled list of raw transcripts
with open('./pickled_files/sotu_raw_transcripts.pkl', 'rb') as f:
    sotu_raw_transcripts = pickle.load(f)
