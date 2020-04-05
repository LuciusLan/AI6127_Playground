#$pythonPath = "C:\ProgramData\Anaconda3\envs\36\python.exe"
$pythonPath = ""
cd $PSScriptRoot
$exec1 = "$pythonPath collect_tweets.py -i dev_offsets.tsv -o tweets.tsv"
$exec2 = "$pythonPath assembleDataFromOffsets.py -t tweets.tsv -i dev_offsets.tsv -o dev_data.tsv -m gold"
$exec3 = "$pythonPath collect_tweets.py -i train_offsets.tsv -o tweets.tsv"
$exec4 = "$pythonPath assembleDataFromOffsets.py -t tweets.tsv -i train_offsets.tsv -o train_data.tsv -m gold"
Invoke-Expression $exec1
Invoke-Expression $exec2
Invoke-Expression $exec3
Invoke-Expression $exec4