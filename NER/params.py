#parameters for the Model
from collections import OrderedDict
import os
parameters = OrderedDict()
parameters['base'] = "C:\\Users\\cifel\\OneDrive\\MSAI\\SourceCodes\\Python\\NER"
parameters['train'] = os.path.join(parameters['base'], ".\\data\\eng.train") #Path to train file
parameters['dev'] = os.path.join(parameters['base'], ".\\data\\eng.testa") #Path to test file
parameters['test'] = os.path.join(parameters['base'], ".\\data\\eng.testb") #Path to dev file
parameters['tag_scheme'] = "BIOES" #BIO or BIOES
parameters['lower'] = True # Boolean variable to control lowercasing of words
parameters['zeros'] = True # Boolean variable to control replacement of  all digits by 0 
parameters['char_dim'] = 30 #Char embedding dimension
parameters['word_dim'] = 100 #Token embedding dimension
parameters['word_lstm_dim'] = 200 #Token LSTM hidden layer size
parameters['word_bidirect'] = True #Use a bidirectional LSTM for words
parameters['embedding_path'] = "D:\\Dev\\Vector\\glove.6B.100d.txt" #Location of pretrained embeddings
parameters['all_emb'] = 1 #Load all embeddings
parameters['crf'] = 1 #Use CRF (0 to disable)
parameters['dropout'] = 0.5 #Droupout on the input (0 = no dropout)
parameters['epoch'] = 10 #Number of epochs to run"
parameters['weights'] = "" #path to Pretrained for from a previous run
parameters['name'] = "self-trained-model_CNN" # Model name
parameters['gradient_clip'] = 5.0
parameters['char_mode'] = "LSTM"