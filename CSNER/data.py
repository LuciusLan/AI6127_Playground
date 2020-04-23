from __future__ import print_function

import torch
import _pickle as cPickle
import os
import codecs
import re
import numpy as np
import gensim
import random
import emoji

from params import parameters

models_path = os.path.join(parameters['base'], "models\\") #path to saved models

#GPU
parameters['use_gpu'] = torch.cuda.is_available() #GPU Check
use_gpu = parameters['use_gpu']

parameters['reload'] = os.path.join(parameters['base'], "models\\self-trained-model")

#Constants
START_TAG = '<START>'
STOP_TAG = '<STOP>'

#paths to files 
#To stored mapping file
mapping_file = os.path.join(parameters['base'], 'data\\mapping.pkl')
train_bin = os.path.join(parameters['base'], 'data\\train.bin')
dev_bin = os.path.join(parameters['base'], 'data\\dev.bin')
test_bin  = os.path.join(parameters['base'], 'data\\test.bin')

#Embedding files
en_emb_path = os.path.join(parameters['embedding_path'], "cc.en.300.vec")
es_emb_path = os.path.join(parameters['embedding_path'], "cc.es.300.vec")
en_bin_path = os.path.join(parameters['embedding_path'], "cc.en.300.bin")
es_bin_path = os.path.join(parameters['embedding_path'], "cc.es.300.bin")

#To stored model
name = parameters['name']
model_name = models_path + name #get_name(parameters)

#if not os.path.exists(models_path):
#    os.makedirs(models_path)


#######################
###                 ###
### Pre Processing  ###
###                 ###
#######################

###             ###
### Data loading###
###             ###

def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub(r'\d', '0', s)

def load_sentences(path, zeros):
    """
    To load the tweets from tsv files. 
    As there is no line separation between tweets, need to add separation step
    Only the word and the NER tag are returned
    """
    collection = []
    prevline = []
    tweet = []
    for line in codecs.open(path, 'r', 'utf-8'):
        line = line.rstrip()
        ls = line.split('\t')
        ls[-2] = pre_process(ls[-2])
        if zeros:               #replace all digits with zero
            ls[-2] = zero_digits(ls[-2])
        if len(prevline) == 0:
            tweet.append(ls[-2:])
            prevline = ls
        if ls[0] == prevline[0]: #Check if two lines belong to same tweet
            tweet.append(ls[-2:])
            prevline = ls
        else: # Separate tweet when tweet ID different
            collection.append(tweet)
            tweet = []
            tweet.append(ls[-2:])
            prevline = ls
    return collection

def pre_process(word):
    """
    To prune the tweet data
    Following same preprocessing pruning as in Wang et. al. paper, i.e.
    • Replaced URLs with [url]
    • Replaced users (starting with @) with [user]
    • Replaced hashtags (starting with # but not
    followed by a number) with [hash tag]
    • Replaced punctuation tokens with [punct]
    • Replaced integer and real numbers by [num]
    • Replaced [num]:[num] with [time]
    • Replaced [num]-[num] with [date]
    • Replaced emojis by [emoji]
    """
    url = re.compile(r'([--:\w?@%&+~#=]*\.[a-z]{2,4}\/{0,2})((?:[?&](?:\w+)=(?:\w+))+|[--:\w?@%&+~#=]+)?')
    username = re.compile(r'^@(?=\S)') #Partial match the at mark, need to replace the whole word when matched
    hashtag = re.compile(r'^#(?=\S)(?!\d)') #Partial match the hash mark
    punct = re.compile(r'\.')
    num = re.compile(r'^\d+.?\d*')
    time = re.compile(r'^\d+\:\d+')
    date = re.compile(r'^\d+\-\d+')
    emojire = re.compile(emoji.get_emoji_regexp()) # The emoji regex is from: https://pypi.org/project/emoji/

    word = url.sub('[url]', word)
    if username.search(word) is not None:
        word = '[user]'
    if hashtag.search(word) is not None:
        word = '[hashtag]'
    word = num.sub('[num]', word)
    word = punct.sub('[punct]', word)
    word = time.sub('[time]', word)
    word = date.sub('[date]', word)
    word = emojire.sub('[emoji]', word)
    return word

def select_test(tweets, n=0.2):
    """
    To randomly pick up test set from original training set. 
    params:
    n: size of testing set ( 0 < n < 1), default 0.2
    """
    l = len(tweets)
    sep = round(n * l)
    random.shuffle(tweets)
    test = tweets[:sep]
    rest = tweets[sep:]
    return test, rest

### Update the tagging scheme,  ###
### change BOIES to standard BOI###

def iob2(tags):
    """
    Check that tags have a valid BIO format.
    Tags in BIO1 format are converted to BIO2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True

def iob_iobes(tags):
    """
    the function is used to convert
    BIO -> BIOES tagging
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
               tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags

def update_tag_scheme(sentences, tag_scheme):
    """
    Check and update sentences tagging scheme to BIO2
    Only BIO1 and BIO2 schemes are accepted for input data.
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        # Check that tags are given in the BIO format
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in BIO format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme == 'BIOES':
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Wrong tagging scheme!')

###                     ###
### Word & char mapping ###
###                     ###

def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert isinstance(item_list, list)
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico

def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item

def word_mapping(sentences, lower):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    words = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    dico = create_dico(words)
    dico['<UNK>'] = 10000000 #UNK tag for unknown words
    word_to_id, id_to_word = create_mapping(dico)
    print("Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in words)
    ))
    return dico, word_to_id, id_to_word

def char_mapping(sentences):
    """
    Create a dictionary and mapping of characters, sorted by frequency.
    """
    chars = ["".join([w[0] for w in s]) for s in sentences]
    dico = create_dico(chars)
    char_to_id, id_to_char = create_mapping(dico)
    print("Found %i unique characters" % len(dico))
    return dico, char_to_id, id_to_char

def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [[word[-1] for word in s] for s in sentences]
    dico = create_dico(tags)
    dico[START_TAG] = -1
    dico[STOP_TAG] = -2
    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found %i unique named entity tags" % len(dico))
    return dico, tag_to_id, id_to_tag

def lower_case(x, lower=False):
    if lower:
        return x.lower()  
    else:
        return x

def prepare_dataset(sentences, word_to_id, char_to_id, tag_to_id, lower=False):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """
    data = []
    for s in sentences:
        str_words = [w[0] for w in s]
        words = [word_to_id[lower_case(w, lower) if lower_case(w, lower) in word_to_id else '<UNK>']
                 for w in str_words]
        # Skip characters that are not in the training set
        chars = [[char_to_id[c] for c in w if c in char_to_id]
                 for w in str_words]
        tags = [tag_to_id[w[-1]] for w in s]
        data.append({
            'str_words': str_words,
            'words': words,
            'chars': chars,
            'tags': tags,
        })
    return data

if parameters['first_time']:
    train_or = load_sentences(parameters['train'], parameters['zeros'])
    test_sentences, train_sentences = select_test(train_or, parameters['test_split'])
    dev_sentences = load_sentences(parameters['dev'], parameters['zeros'])

    dico_words, word_to_id, id_to_word = word_mapping(train_sentences, parameters['lower'])
    dico_chars, char_to_id, id_to_char = char_mapping(train_sentences)
    dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)

    train_data = prepare_dataset(
        train_sentences, word_to_id, char_to_id, tag_to_id, parameters['lower']
    )
    dev_data = prepare_dataset(
        dev_sentences, word_to_id, char_to_id, tag_to_id, parameters['lower']
    )
    test_data = prepare_dataset(
        test_sentences, word_to_id, char_to_id, tag_to_id, parameters['lower']
    )

    print("{} / {} / {} sentences in train / dev / test.".format(
        len(train_data), len(dev_data), len(test_data)))

    #######################
    ###                 ###
    ###  Word Embedding ###
    ###                 ###
    #######################
    
    """
    all_word_embeds = {}
    for i, line in enumerate(codecs.open(parameters['embedding_path'], 'r', 'utf-8')):
        s = line.strip().split()
        if len(s) == parameters['word_dim'] + 1:
            all_word_embeds[s[0]] = np.array([float(i) for i in s[1:]])
    """

    def update_dataset(dataset, lang, emb):
        if lang.lower() not in ['en', 'es']:
            raise ValueError("Lang must be either En or Es")
        if lang.lower() == 'en':
            for tweet in dataset:
                flagen = []
                for word in tweet['str_words']:
                    flagen.append(word in en_emb)
                new_tweet = {
                    **tweet,
                    'en_flag': flagen
                }
                tweet.update(new_tweet)
        elif lang.lower() == 'es':
            for tweet in train_data:
                flages = []
                for word in tweet['str_words']:
                    flages.append(word in es_emb)
                new_tweet = {
                    **tweet,
                    'es_flag': flages
                }
                tweet.update(new_tweet)

    #Intializing Word Embedding Matrix
    en_emb = gensim.models.KeyedVectors.load_word2vec_format(fname=en_bin_path, binary=True)
    #en_emb.save_word2vec_format(os.path.join(parameters['embedding_path'], 'cc.en.300.bin'), binary=True)
    en_word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(word_to_id), parameters['word_dim']))
    
    for w in word_to_id:
        if w in en_emb:
            en_word_embeds[word_to_id[w]] = en_emb[w]
        elif w.lower() in en_emb:
            en_word_embeds[word_to_id[w]] = en_emb[w.lower()]
    print('Loaded %i pretrained embeddings (English).' % len(en_emb.vocab))
    
    update_dataset(train_data, 'en', en_emb)
    update_dataset(test_data, 'en', en_emb)
    update_dataset(dev_data, 'en', en_emb)

    del en_emb

    es_emb = gensim.models.KeyedVectors.load_word2vec_format(es_bin_path, binary=True)
    #es_emb.save_word2vec_format(os.path.join(parameters['embedding_path'], 'cc.es.300.bin'), binary=True)
    es_word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(word_to_id), parameters['word_dim']))

    for w in word_to_id:
        if w in es_emb:
            es_word_embeds[word_to_id[w]] = es_emb[w]
        elif w.lower() in es_emb:
            es_word_embeds[word_to_id[w]] = es_emb[w.lower()]
    print('Loaded %i pretrained embeddings (Spanish).' % len(es_emb.vocab))

    update_dataset(train_data, 'es', es_emb)
    update_dataset(test_data, 'es', es_emb)
    update_dataset(dev_data, 'es', es_emb)

    del es_emb

    with open(train_bin, 'wb') as f:
        cPickle.dump(train_data, f)
    with open(dev_bin, 'wb') as f:
        cPickle.dump(dev_data, f)
    with open(test_bin, 'wb') as f:
        cPickle.dump(test_data, f)
    

    with open(mapping_file, 'wb') as f:
        mappings = {
            'word_to_id': word_to_id,
            'tag_to_id': tag_to_id,
            'char_to_id': char_to_id,
            'en_word_embeds': en_word_embeds,
            'es_word_embeds': es_word_embeds
        }
        cPickle.dump(mappings, f)

    print('word_to_id: ', len(word_to_id))
else:
    # After first execution, directly load saved binaries
    with open(mapping_file, 'rb') as f:
        t = cPickle.load(f)
        word_to_id = t['word_to_id']
        tag_to_id = t['tag_to_id']
        char_to_id = t['char_to_id']
        en_word_embeds = t['en_word_embeds']
        es_word_embeds = t['es_word_embeds']
    
    with open(train_bin, 'rb') as f:
        train_data = cPickle.load(f)

    with open(dev_bin, 'rb') as f:
        dev_data = cPickle.load(f)

    with open(test_bin, 'rb') as f:
        test_data = cPickle.load(f)
