python3 collect_tweets.py -i ./dev_offsets.tsv -o ./tweets.tsv
python3 assembleDataFromOffsets.py -t ./tweets.tsv -i ./dev_offsets.tsv -o ./dev_data.tsv -m gold
python3 collect_tweets.py -i ./train_offsets.tsv -o ./tweets.tsv
python3 assembleDataFromOffsets.py -t ./tweets.tsv -i ./train_offsets.tsv -o ./train_data.tsv -m gold