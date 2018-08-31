# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 15:23:24 2018

@author: zyv57124
"""

from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.keys import Keys
import time
import urllib
import cStringIO
import re
import requests
from bs4 import BeautifulSoup as BS

browser = webdriver.Chrome(r'C:\Users\zyv57124\AppData\Local\Continuum\miniconda3\envs\images\Library\bin\chromedriver.exe')

browser.get('https://images.google.co.uk')

search = browser.find_element_by_name('q')


label = "Brass bronze"
search_term = "microstructure"

search_term = label+search_term


search.send_keys(search_term)

search.send_keys(Keys.RETURN)

SCROLL_PAUSE_TIME = 5.0

# Get scroll height
last_height = browser.execute_script("return document.body.scrollHeight")

while True:
    # Scroll down to bottom
    browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    # Wait to load page
    time.sleep(SCROLL_PAUSE_TIME)
    urls = []
        
    images = browser.find_elements_by_tag_name('img')
    
    for i in images:
        
        urls.append(i.get_attribute("src"))
        
        urls = [i for i in urls if i is not None]
    
    # Calculate new scroll height and compare with last scroll height
    new_height = browser.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height
    
    while True:
        try:
            loadMoreButton = browser.find_element_by_xpath("//button[normalize-space()='show more results']")
            loadMoreButton.click()
        except Exception as e:
            print e
            break

for each in range(len(urls)):
        
        urllib.urlretrieve(urls[each], label+str(each)+".png")

print len(urls)
print "Complete"