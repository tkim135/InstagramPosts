#!/usr/bin/env python

import instaloader
from langdetect import detect
from datetime import datetime, timedelta

# Get instance
L = instaloader.Instaloader()

profiles = []

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
						#scrape posts on profile
			except langdetect.lang_detect_exception.LangDetectException:
				continue



