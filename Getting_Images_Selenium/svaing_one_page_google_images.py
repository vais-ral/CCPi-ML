# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 16:57:08 2018

@author: zyv57124
"""
#C:\Users\zyv57124\AppData\Local\Continuum\miniconda3\envs\images\Library\bin\chromedriver.exe n    path to chrome driver

from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.keys import Keys
import time
import urllib
import cStringIO
import re
import requests
from bs4 import BeautifulSoup as BS

browser = webdriver.Chrome(r'C:\Users\zyv57124\AppData\Local\Continuum\miniconda3\envs\images\Library\bin\chromedriver.exe')

browser.get('http://images.google.co.uk')

search = browser.find_element_by_name('q')


label = "spiral"
search_term = label+" galaxy astronomy telescope"


search.send_keys(search_term)

search.send_keys(Keys.RETURN)

browser.save_screenshot("screenshot.png")

urls = []
        
images = browser.find_elements_by_tag_name('img')
    
for i in images:
        
    urls.append(i.get_attribute("src"))
        
urls = [i for i in urls if i is not None]
    
for each in range(len(urls)):
        
    urllib.urlretrieve(urls[each], label+str(each)+".png")