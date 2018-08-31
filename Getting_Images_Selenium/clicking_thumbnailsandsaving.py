# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 15:22:57 2018

@author: zyv57124
"""

#<a href="search.php?q=grey+cast+iron&amp;page=3" title="Go to next page of results">Next</a>

from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.keys import Keys
import time
import urllib
import cStringIO
import re
import requests
from bs4 import BeautifulSoup as BS
import numpy as np

browser = webdriver.Chrome(r'C:\Users\zyv57124\AppData\Local\Continuum\miniconda3\envs\images\Library\bin\chromedriver.exe')

browser.get('https://www.doitpoms.ac.uk/miclib/search.php?q=brass&%24layout_type=1&action=Go')

label = "brass"

urls = []
    
# grab the data
while True:
     time.sleep(4)
#    images = browser.find_elements_by_tag_name('img')
#    
#    for i in images:
#        
#        urls.append(i.get_attribute("src"))
#        
#        urls = [i for i in urls if i is not None]
#        
#        numbers = np.arange(0,len(urls), 1)
#        
#    for number in numbers:
#        image = urls[number]
     pic = browser.find_element_by_xpath('//*[@alt="Link to full record for micrograph 29"]')
     pic.click
    # click next link
     time.sleep(5)  
    
    #elm = browser.find_element_by_xpath('//*[@title="Go to next page of results"]')
    
    #elm.click()

for each in range(len(urls)):
    
    urllib.urlretrieve(urls[each], label+"test"+str(each)+".png")

print len(urls)

print "Complete"