# -*- coding: utf-8 -*-

import argparse
import tweepy
import time
import sys
import os
# TWITTER_URL = "https://twitter.com/"

base = os.path.dirname(os.path.abspath(__file__))
twitter_auth_file = os.path.join(base, "twitter_auth.txt")


def get_tweets(twitter, tweet_ids):
    """fetch the tweet from given tweet_id
            returns tweet text if found, otherwise returns Not Found
    """
    # url = TWITTER_URL + user_id + "/status/" + tweet_id
    # tweet = 'Not Found'
    new_tweets = []
    success = False

    while not success:
        try:
            result = twitter.statuses_lookup(
                tweet_ids)  # Get tweets from Twitter

            for twit in result:
                tweet_id = twit.id_str
                user_id = twit.user.id_str
                tweet_text = twit.text.replace("\n", " ")
                tweet_text = tweet_text.replace("\r", " ")
                tweet_text = tweet_text.replace(u"&lt;", u"<")
                tweet_text = tweet_text.replace(u"&gt;", u">")
                tweet_text = tweet_text.replace(u"&amp;", u"&")
                new_tweets.append([tweet_id, user_id, tweet_text])
            # print("X", end="")
            success = True

        except tweepy.RateLimitError as e:  # Hit Twitter limit, wait 15 minutes
            print()
            print("Hit Twitter Limit, waiting 15 minutes.")
            time.sleep(15 * 60 + 15)
            print("Resume...")
            success = False

        except tweepy.TweepError as e:  # Possibly not found or unauthorized
            print()
            # print('Error getting ', " tweet id: ", tweet_id)
            print('Reason: ', e.reason)
            with open("msg.txt", "a", encoding='utf-8') as out:
                out.write('Reason: ' + e.reason + "\n")
            print("\nWait 2 min...\n")
            time.sleep(2 * 60 + 15)
            print("Resume...")
            success = False
    return new_tweets


if __name__ == "__main__":
    # configuraion for parsing command line arguments
    parser = argparse.ArgumentParser("usage: %prog [options] ")
    parser.add_argument("-i", "--input", dest="input_file",
                      type="string", help="specify input filename")
    parser.add_argument("-o", "--output", dest="output_file",
                      type="string", help="specify output filename")
    opts = parser.parse_args()
    if opts.input_file is None or opts.output_file is None:
        parser.print_help()
        parser.error(" ")
    else:
        input_file = os.path.join(base, opts.input_file)
        output_file = os.path.join(base, opts.output_file)

        # Instantiate api connection #######################
        with open(twitter_auth_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            try:
                consumer_key = lines[0].strip()
                consumer_secret = lines[1].strip()
                access_key = lines[2].strip()
                access_secret = lines[3].strip()
            except IndexError as e:
                print("Please fill authorization file!")
                sys.exit()

        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_key, access_secret)
        twitter = tweepy.API(auth)
        ####################################################

        print("Fetching tweets ")
        time_start = time.time()

        # open input file to read the tweet id and user id
        with open(input_file) as f:
            # read tweet_ids and store in a list
            old_tweet_id, old_user_id = '', ''
            tweet_ids = []
            for line in f:
                if line.strip():  # if empty line skip
                    try:
                        tweet_id, user_id = line.strip().split('\t')[0:2]
                        if not (old_tweet_id == tweet_id and old_user_id == user_id):
                            tweet_ids.append(tweet_id)
                            # out.write(tweet_id + '\t' + user_id + '\t' + tweet + "\n")
                            # out.flush()
                            # print("+", end=' ')
                        old_tweet_id, old_user_id = tweet_id, user_id
                    except ValueError as e:
                        print("Value Error for: "+line+"  :" + str(e))

        # fetch tweets and write tweets to file
        tweets = []
        print("Tweets left: " + str(len(tweet_ids)))
        while len(tweet_ids) > 0:
            tweet_slice = tweet_ids[:100]
            tweet_ids = tweet_ids[100:]
            tweets.extend(get_tweets(twitter, tweet_slice))
            print("Tweets left: " + str(len(tweet_ids)))

        with open(output_file, 'w', encoding='utf-8') as out:
            for tweet in tweets:
                out.write(tweet[0] + "\t" + tweet[1] + "\t" + tweet[2] + "\n")

        print()
        time_end = time.time()
        print("Done")
        print("Collection time: " + str(time_end - time_start) + " seconds.")
        with open("msg.txt", "a", encoding='utf-8') as out:
            out.write("Collection time: " +
                      str(time_end - time_start) + " seconds.\n")
