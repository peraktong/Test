import bs4
import requests
import re
import os
url = "http://pages.stat.wisc.edu/~wardrop/courses/"
r = requests.get(url)
data = bs4.BeautifulSoup(r.text, "html.parser")
count=0
for l in data.find_all("a"):
    r = requests.get(url + l["href"])
    print(r.status_code,"%d of %d"%(count,len(data.find_all("a"))))
    open('%s.pdf' % count, 'wb').write(r.content)
    count+=1
