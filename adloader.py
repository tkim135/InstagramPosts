#!/usr/bin/env python

import csv
import instaloader
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
                            #scrape posts on profile
                            postNum += 1
                            row = [postNum, post.date_local, post.caption, 
                                post.url, post.tagged_users, post.likes, post.comments, post.location]
                            writer.writerow(row)
                            
                except langdetect.lang_detect_exception.LangDetectException:
                    continue

csvFile.close()

