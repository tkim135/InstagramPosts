#!/usr/bin/env python

import csv
import instaloader
import langdetect
import urllib.request
from langdetect import detect
from datetime import datetime, timedelta

# Get instance
L = instaloader.Instaloader()
# L.interactive_login('mbbackus')
csvFile = "post-data.csv"

profiles = []

with open(csvFile, 'a') as csvFile:
    writer = csv.writer(csvFile)
    postNum = 0
    # try: 
    for post in L.get_hashtag_posts('ad'):
        if post.likes < 1000:
            continue
        if type(post.caption) != str:
            continue
        if not post.caption:
            continue
        try:
            if not detect(post.caption) == 'en':
                continue
            profile = post.owner_profile
            if profile in profiles:
                continue
            profiles.append(profile)

            print(profile)

            for nupost in profile.get_posts():
                if detect(nupost.caption) != 'en':
                    continue
                if nupost.date_local.year != 2019:
                    break
                postNum += 1
                imagename = "image{}.jpg".format(postNum)
                urllib.request.urlretrieve(nupost.url, "img/{}".format(imagename))
                row = [postNum, nupost.date_local, nupost.caption, 
                    imagename, nupost.tagged_users, nupost.likes, 
                    nupost.comments, nupost.location]
                writer.writerow(row)
                    
        except langdetect.lang_detect_exception.LangDetectException:
            continue
    # except Exception as e:
    #     print('Encountered exception at {0} with the following error message: {1}'.format(str(datetime.now()), str(e)))

csvFile.close()

