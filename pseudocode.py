#!/usr/bin/env python

import csv
import instaloader
import langdetect
import threading
import urllib.request
from langdetect import detect
from datetime import datetime, timedelta

csvFile = "post-data.csv"
s = threading.Semaphore(4)
l = threading.Lock()
profiles = []

def get_posts(profile):
    print(profile)
    postNum = 0
    for nupost in profile.get_posts():
        if detect(nupost.caption) != 'en':
            continue
        if nupost.date_local.year != 2019:
            s.release()
            return
        imagename = "%s%d.jpg" % (profile.username, postNum)
        urllib.request.urlretrieve(nupost.url, "img/{}".format(imagename))
        row = [postNum, nupost.date_local, nupost_caption, imagename, nupost.tagged_users, nupost.likes, nupost.comments, nupost.location]
        l.acquire()
        writer.writerow(row)
        l.release()
        postNum += 1

L = instaloader.Instaloader()

with open(csvFile, 'a') as csvFile:
    writer = csv.writer(csvFile)
    try:
        for post in L.get_hashtag_posts('ad'):
            if post.likes < 1000 or type(post.caption) != str or not post.caption:
                continue
            try:
                if not detect(post.caption) == 'en':
                    continue
                print(post.caption)
                print(post.owner_profile)
                profile = post.owner_profile
                print("and also here")
                if profile in profiles:
                    continue
                print("checked")
                profiles.append(profile)
                print("appended")
                S.acquire()
                t = Thread(target=get_posts, args=(profile,))
                t.start()
            except langdetect.lang_detect_exception.LangDetectException:
                continue
    except Exception as e:
        print('Encountered exception at {0} with the following error message: {1}'.format(str(datetime.now()), str(e)))

csvFile.close()
