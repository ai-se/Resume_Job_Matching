import numpy as np
import pandas as pd
import re
import sys
import requests
from bs4 import BeautifulSoup as bs
links = []
usernames = []
jobtitle = sys.argv[1] if len(sys.argv) >= 2 else 'engineer'
start = 1
end = int(sys.argv[2]) if len(sys.argv) >= 3 else 5
filename = sys.argv[3] if len(sys.argv) >= 4 else "resumesE.csv"
type = sys.argv[4] if len(sys.argv) == 5 else "resumes"
for i in range(start, end+1):
    url = 'https://www.postjobfree.com/'+type+'?q='+jobtitle+'&l=&radius=25&r=100&p='+str(i)
    r = requests.get(url)
    soup = bs(r.content, "html.parser")
    mydivs = soup.findAll("h3", {"class": "itemTitle"})
    for div in mydivs:
        url = 'https://www.postjobfree.com' + (div.find("a")["href"])
        links.append(url)
        usernames.append(url.split('/')[4])

with open(filename, "w") as f:
    f.write("links,usernames,\n")
    for link, username in zip(links, usernames):
        f.write(link + ","+username+",\n")

