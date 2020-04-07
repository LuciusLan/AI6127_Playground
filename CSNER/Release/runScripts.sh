python3 collect_tweets.py -i $1 -o tweets.tsv
python3 assembleDataFromOffsets.py -t tweets.tsv -i $1 -o data.tsv -m $2
