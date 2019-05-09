from twitter_scraper import get_tweets

for tweet in get_tweets('KonyTim946', pages=1):
	print('------------------')
	print(tweet['text'])
	print('------------------')
