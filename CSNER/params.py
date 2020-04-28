#parameters for the Model
from collections import OrderedDict
import os
parameters = OrderedDict()

parameters['first_time'] = False


parameters['base'] = os.path.dirname(os.path.abspath(__file__))
# Windows style dir
parameters['train'] = os.path.join(parameters['base'], "data\\train_data.tsv") #Path to train file
parameters['dev'] = os.path.join(parameters['base'], "data\\dev_data.tsv") #Path to dev file
parameters['embedding_path'] = "D:\\Dev\\Vector\\" #Location of pretrained embeddings
parameters['test_split'] = 0.2
# *nix style dir
#parameters['train'] = os.path.join(parameters['base'], "data/train_data.tsv") #Path to train file
#parameters['dev'] = os.path.join(parameters['base'], "data/dev_data.tsv") #Path to test file
#parameters['embedding_path'] = "/mnt/d/Dev/Vector" #Location of pretrained embeddings

parameters['tag_scheme'] = "BIO" #BIO or BIOES
parameters['lower'] = True # Boolean variable to control lowercasing of words
parameters['zeros'] = False # Boolean variable to control replacement of  all digits by 0 
parameters['char_dim'] = 30 #Char embedding dimension
parameters['word_dim'] = 300 #Token embedding dimension
parameters['word_lstm_dim'] = 600 #Token LSTM hidden layer size
parameters['word_bidirect'] = True #Use a bidirectional LSTM for words

parameters['all_emb'] = 1 #Load all embeddings
parameters['crf'] = 1 #Use CRF (0 to disable)
parameters['dropout'] = 0.5 #Droupout on the input (0 = no dropout)
parameters['epoch'] = 100 #Number of epochs to run"
parameters['weights'] = "" #path to Pretrained for from a previous run
parameters['batch_size'] = 64
parameters['name'] = "Baseline_SimpleConcat" # Model name
parameters['gradient_clip'] = 5.0
parameters['char_mode'] = "LSTM"
parameters['start_type'] = "cold"
parameters['early_stop_thres'] = 3
parameters['attention'] = 'no_dep_softmax' # None, dep_softmax, no_dep_softmax