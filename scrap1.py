import requests
from bs4 import BeautifulSoup

url = 'https://coinmarketcap.com/trending-cryptocurrencies'
response = requests.get(url)

soup = BeautifulSoup(response.content, 'html.parser')
data = soup.get_text(strip=True)

with open('bdo.txt', 'w') as f:
    f.write(data)