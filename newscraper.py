#!/usr/bin/env python

import csv
import instaloader
import langdetect
import threading
import sys
from time import sleep
import urllib.request
from langdetect import detect
from datetime import datetime, timedelta
from PIL import Image
import requests
from io import BytesIO

csvFile = "post-data.csv"
s = threading.Semaphore(10)
l = threading.Lock()
p = threading.Lock()
profiles = []



def isAd(hashtags, caption):
    return '[ad]' in caption or 'ad' in hashtags or 'sponsored' in hashtags or 'advertisement' in hashtags or 'spon' in hashtags 

def get_posts(profile):
    p.acquire()
    print(profile)
    p.release()
    postNum = 0
    adCount = 0
    for nupost in profile.get_posts():
        try:
            if not nupost.caption or detect(nupost.caption) != 'en':
                continue
            if nupost.date_local.year != 2019:
                return
            adbool = isAd(nupost.caption_hashtags, nupost.caption);
            if not adbool and adCount < postNum / 2:
                continue
            imagename = "%s%d.jpg" % (profile.username, postNum)
            #urllib.request.urlretrieve(nupost.url, "img/{}".format(imagename))
            response = requests.get(nupost.url)
            img = Image.open(BytesIO(response.content))
            width, height = img.size
            if width >= 1080 and height >= 1080:
                topy = (height - 1080) / 2
                topx = (width - 1080) / 2
                boty = 1080 + topy
                botx = 1080 + topx
                box = (topx, topy, botx, boty)
                newimg = img.crop(box)
                newimg = newimg.resize((512, 512))
                newimg.save("img/{}".format(imagename))
            else:
                img = img.resize((512,512))
                img.save("img/{}".format(imagename))
            row = [postNum, nupost.date_local, nupost.caption.replace("\n",""), imagename, nupost.tagged_users, nupost.likes, nupost.comments, int(adbool)]#, nupost.location]
            l.acquire()
            writer.writerow(row)
            l.release()
            if adbool:
                adCount += 1
            postNum += 1
        except langdetect.lang_detect_exception.LangDetectException:
            continue
        except TypeError:
            with p:
                print(nupost.caption)
            continue

L = instaloader.Instaloader()
thread_list = []



with open(csvFile, 'a') as csvFile:
    writer = csv.writer(csvFile)
    while True:
        try:
            for post in L.get_hashtag_posts('ad'):
                if post.likes < 1000 or type(post.caption) != str or not post.caption:
                    continue
                try:
                    if not detect(post.caption) == 'en':
                        continue
                    profile = post.owner_profile
                    if profile in profiles:
                        continue
                    profiles.append(profile)
                    with s:
                        t = threading.Thread(target=get_posts, args=(profile,))
                        thread_list.append(t)
                        t.start()
                    sys.stdout.flush()
                except langdetect.lang_detect_exception.LangDetectException:
                    continue
        except Exception as e:
            with p:
                print('Encountered exception at {0} with the following error message: {1}'.format(str(datetime.now()), str(e)))
            if str(e) == 'query_hash':
                sleep(10)

for thread in thread_list:
    thread.join()
csvFile.close()
