#!/usr/bin/env python

import csv
import instaloader
import langdetect
import urllib.request
from langdetect import detect
from datetime import datetime, timedelta

# Get instance
L = instaloader.Instaloader()
csvFile = "post-data.csv"

profiles = []

with open(csvFile, 'a') as csvFile:
    writer = csv.writer(csvFile)
    postNum = 0
    for post in L.get_hashtag_posts('ad'):
        if post.likes >= 1000:
            if type(post.caption) != str:
                continue
            if post.caption:
                try:
                    if detect(post.caption) == 'en':
                        profile = post.owner_profile
                        if profile not in profiles:
                            profiles.append(profile)
                            print(profile)
                            for nupost in profile.get_posts():
                            	postNum += 1
                            	imagename = "image{}.jpg".format(postNum)
                            	urllib.request.urlretrieve(nupost.url, "img/{}".format(imagename))
                            	row = [postNum, nupost.date_local, nupost.caption, 
                                	imagename, nupost.tagged_users, nupost.likes, 
                                	nupost.comments, nupost.location]
                            	writer.writerow(row)
                            
                except langdetect.lang_detect_exception.LangDetectException:
                    continue

csvFile.close()

