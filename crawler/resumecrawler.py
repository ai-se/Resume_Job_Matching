import numpy as np
import pandas as pd
import re
import time
import sys
import requests
from bs4 import BeautifulSoup as bs

df = pd.read_csv("jobsMO.csv")

for i, url in enumerate(df.links):
    time.sleep(1)
    filename = url.split('/')[4]
    print(filename)
    with open('./jobs/'+filename+'.txt', 'w') as f:
        print("scraping number", i, "url = ",  url)
        r = requests.get(url)
        soup = bs(r.content, "html.parser")
        mydivs = soup.findAll("div", {"class": "normalText"})
        for div in mydivs:
            f.write(div.text)