# This script will get the tokens from the tweets using the provided offsets.
# Requires a raw tweet file with columns tweet_id, user_id tweet_text
# Requires an offset file with format tweet_id, user_id, start, end, [annotation]
# There are two modes to run this script, Regular and Gold
# Gold mode will include Gold standard annotation as the last column, regular will not.

import argparse
import codecs
import os

base = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    # configuraion for parsing command line arguments
    parser = argparse.ArgumentParser("usage: %prog [options] ")
    parser.add_argument("-t", "--tweets", dest="tweet_file", type=str, help="specify tweet filename")
    parser.add_argument("-i", "--input", dest="offset_file", type=str, help="specify offset filename")
    parser.add_argument("-o", "--output", dest="output_file", type=str, help="specify output filename")
    parser.add_argument("-m", "--mode", dest="mode", type=str, help="specify mode (reg|gold)")
    opts = parser.parse_args()
    if opts.offset_file is None or opts.output_file is None or opts.tweet_file is None or opts.mode is None:
        parser.print_help()
        parser.error(" ")
    elif opts.mode != "reg" and opts.mode != "gold":
        parser.print_help()
        parser.error("Incorrect mode")
    else:
        offset_file = os.path.join(base, opts.offset_file)
        output_file = os.path.join(base, opts.output_file)
        tweet_file = os.path.join(base, opts.tweet_file)
        mode = opts.mode

    # Read tweet file and store into Dictionary
    tweets = {}
    with open(tweet_file, 'r', encoding='utf-8') as tf:
        for line in tf:
            if line.strip():
                try:
                    tweet_id, user_id, tweet_text = line.strip().split("\t")
                    if tweet_id not in tweets:
                        tweets[tweet_id] = tweet_text
                    else:
                        print("Tweet exists.")
                except ValueError as e:
                    print(tweet_id)

    # Read offset file and add token
    with open(offset_file, 'r', encoding='utf-8') as offset:
        out = codecs.open(output_file, 'w', encoding='utf-8')
        # nf = codecs.open(output_file+"_nf", 'w', encoding='utf-8')
        cols = len(offset.readline().strip().split("\t"))
        offset.seek(0)
        for line in offset:
            if line.strip():  # Skip empty lines
                try:
                    if cols == 4:
                        tweet_id, user_id, start, end = line.strip().split("\t")
                        gold = ""
                    elif cols == 5:
                        tweet_id, user_id, start, end, gold = line.strip().split("\t")
                    elif cols == 6:
                        tweet_id, user_id, start, end, old_token, gold = line.strip().split("\t")

                    # Get token
                    token = ""
                    if tweet_id in tweets:
                        token = tweets[tweet_id][int(start):int(end)+1]
                    else:
                        print("Tweet doesn't exist: " + tweet_id)
                        # nf.write(tweet_id+"\t"+user_id+"\t"+start+"\t"+end+"\n")
                        continue

                    # Write to file
                    if mode == "reg":
                        text = tweet_id+"\t"+user_id+"\t"+start+"\t"+end+"\t"+token
                    elif mode == "gold":
                        text = tweet_id + "\t" + user_id + "\t" + \
                            start + "\t" + end + "\t" + token + "\t" + gold
                    out.write(text+"\n")
                except ValueError as e:
                    print("Value Error for: "+line+"  :"+ str(e))
        out.close()
        print("Done.")
        # nf.close()
