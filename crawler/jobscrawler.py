import numpy as np
import pandas as pd
import re
import time
import sys
import requests
from bs4 import BeautifulSoup as bs

df = pd.read_csv("jobsE.csv")


for i, url in enumerate(df.links):
    time.sleep(1)
    filename = url.split('/')[4]
    print(filename)
    with open('./jobs/'+filename+'.txt', 'w') as f:
        print("scraping number", i, "url = ",  url)
        r = requests.get(url)
        soup = bs(r.content, "html.parser")
        title = soup.find("h1").text
        print(title)
        f.write("Job Title = " + title + "\n")
        mydivs = soup.findAll("div", {"class": "normalText"})
        for div in mydivs:
            f.write(div.text)