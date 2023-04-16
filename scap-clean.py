import requests
from bs4 import BeautifulSoup
import re

url = 'https://coinmarketcap.com/trending-cryptocurrencies'
response = requests.get(url)

soup = BeautifulSoup(response.content, 'html.parser')
data = soup.get_text(strip=True)

# Remove special characters and extra whitespace
clean_data = re.sub('[^0-9a-zA-Z\s]+', '', data).strip()

# Split the text into lines
lines = clean_data.split('\n')

# Remove empty lines
lines = [line for line in lines if line]

# Write the cleaned data to a text file
with open('one.doc', 'w') as f:
    f.write('\n'.join(lines))