import tweepy
import pandas as pd

# Twitter API credentials
consumer_key = "ISCUHiifzDRmdfKDoVFoxp5qh"
consumer_secret = "SnsePD0F9amOuwGY6fA4Baw2vChVnpjDWbHwyRLYAwAzjEEHa9"
access_token = "1454169999413846016-ZAzL5nc4hCzOyV3kj9ZMjB3mzG5JhI"
access_token_secret = "kA4ezPPhnrxNq2vPEAk6MdplgpR83zZtjRxHW8RQIIJOV"

# Set up authentication with Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# Create API object
api = tweepy.API(auth)

# Define search query and number of tweets to retrieve
query = ""
count = 100

# Retrieve tweets
tweets = tweepy.Cursor(api.search_tweets,
                       q=query,
                       lang="en").items(count)

# Convert tweets to a pandas dataframe
data = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweet'])

# Save data to Excel file
data.to_excel("tweets.xlsx", index=False)




import snscrape.modules.twitter as sntwitter
import pandas as pd

query = "python"
tweets = []
limit = 100


for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    
    # print(vars(tweet))
    # break
    if len(tweets) == limit:
        break
    else:
        tweets.append([tweet.date, tweet.username, tweet.content])
        
df = pd.DataFrame(tweets, columns=['Date', 'User', 'Tweet'])
print(df)

# to save to csv
 df.to_csv('tweets.csv')