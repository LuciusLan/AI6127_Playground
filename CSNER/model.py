import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch import autograd
from torch import Tensor
import torch
from params import parameters
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

START_TAG = '<START>'
STOP_TAG = '<STOP>'
PAD_TAG = '<PAD>'

###########################
###########################
### Initialization and  ###
### util functions      ###
###########################


def init_embedding(input_embedding):
    """
    Initialize embedding
    """
    bias = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform(input_embedding, -bias, bias)


def init_linear(input_linear):
    """
    Initialize linear transformation
    """
    bias = np.sqrt(6.0 / (input_linear.weight.size(0) +
                          input_linear.weight.size(1)))
    nn.init.uniform(input_linear.weight, -bias, bias)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()


def init_lstm(input_lstm):
    """
    Initialize lstm

    PyTorch weights parameters:

        weight_ih_l[k]: the learnable input-hidden weights of the k-th layer,
            of shape `(hidden_size * input_size)` for `k = 0`. Otherwise, the shape is
            `(hidden_size * hidden_size)`

        weight_hh_l[k]: the learnable hidden-hidden weights of the k-th layer,
            of shape `(hidden_size * hidden_size)`            
    """

    # Weights init for forward layer
    for ind in range(0, input_lstm.num_layers):

        # Gets the weights Tensor from our model, for the input-hidden weights in our current layer
        weight = eval('input_lstm.weight_ih_l' + str(ind))

        # Initialize the sampling range
        sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))

        # Randomly sample from our samping range using uniform distribution and apply it to our current layer
        #nn.init.uniform(weight, -sampling_range, sampling_range)
        nn.init.orthogonal_(weight)
        # Similar to above but for the hidden-hidden weights of the current layer
        weight = eval('input_lstm.weight_hh_l' + str(ind))
        sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        #nn.init.uniform(weight, -sampling_range, sampling_range)
        nn.init.orthogonal_(weight)

    # We do the above again, for the backward layer if we are using a bi-directional LSTM (our final model uses this)
    if input_lstm.bidirectional:
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.weight_ih_l' + str(ind) + '_reverse')
            sampling_range = np.sqrt(
                6.0 / (weight.size(0) / 4 + weight.size(1)))
            #nn.init.uniform(weight, -sampling_range, sampling_range)
            nn.init.orthogonal_(weight)
            weight = eval('input_lstm.weight_hh_l' + str(ind) + '_reverse')
            sampling_range = np.sqrt(
                6.0 / (weight.size(0) / 4 + weight.size(1)))
            #nn.init.uniform(weight, -sampling_range, sampling_range)
            nn.init.orthogonal_(weight)

    # Bias initialization steps

    # We initialize them to zero except for the forget gate bias, which is initialized to 1
    if input_lstm.bias:
        for ind in range(0, input_lstm.num_layers):
            bias = eval('input_lstm.bias_ih_l' + str(ind))

            # Initializing to zero
            bias.data.zero_()

            # This is the range of indices for our forget gates for each LSTM cell
            bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1

            # Similar for the hidden-hidden layer
            bias = eval('input_lstm.bias_hh_l' + str(ind))
            bias.data.zero_()
            bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1

        # Similar to above, we do for backward layer if we are using a bi-directional LSTM
        if input_lstm.bidirectional:
            for ind in range(0, input_lstm.num_layers):
                bias = eval('input_lstm.bias_ih_l' + str(ind) + '_reverse')
                bias.data.zero_()
                bias.data[input_lstm.hidden_size: 2 *
                          input_lstm.hidden_size] = 1
                bias = eval('input_lstm.bias_hh_l' + str(ind) + '_reverse')
                bias.data.zero_()
                bias.data[input_lstm.hidden_size: 2 *
                          input_lstm.hidden_size] = 1


def log_sum_exp(vec):
    """
    Calculate the log_sum_exp trick for the tensor.
    :param vec: [batchSize * from_label * to_label].
    :return: [batchSize * to_label]
    """
    maxScores, idx = torch.max(vec, 1)
    maxScores[maxScores == -float("Inf")] = 0
    maxScoresExpanded = maxScores.view(vec.shape[0], 1, vec.shape[2]).expand(
        vec.shape[0], vec.shape[1], vec.shape[2])
    return maxScores + torch.log(torch.sum(torch.exp(vec - maxScoresExpanded), 1))


def argmax(vec):
    '''
    This function returns the max index in a vector
    '''
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def to_scalar(var):
    '''
    Function to convert pytorch tensor to a scalar
    '''
    return var.view(-1).data.tolist()[0]


def score_sentences(self, feats: Tensor, tags: Tensor):
    # tags is ground_truth, a list of ints, length is len(sentence)
    # feats is a 2D tensor, len(sentence) * tagset_size

    # feats and tags are padded with zero to fit longest sequence in batch
    # need to prune here to avoid extremly small value
    mask = tags.ne(0.)
    tags = torch.masked_select(tags, mask)
    r = torch.LongTensor(range(tags.size()[0]))
    if self.use_gpu:
        r = r.cuda()
        pad_start_tags = torch.cat(
            [torch.cuda.LongTensor([self.tag_to_ix[START_TAG]]), tags])
        pad_stop_tags = torch.cat(
            [tags, torch.cuda.LongTensor([self.tag_to_ix[STOP_TAG]])])
    else:
        pad_start_tags = torch.cat(
            [torch.LongTensor([self.tag_to_ix[START_TAG]]), tags])
        pad_stop_tags = torch.cat(
            [tags, torch.LongTensor([self.tag_to_ix[STOP_TAG]])])

    score = torch.sum(
        self.transitions[pad_stop_tags, pad_start_tags]) + torch.sum(feats[r, tags])

    return score


def forward_pred(self, feats: Tensor, batch_sent_length: Tensor):
    '''
    Calculates the forwarded prediction score (NLL)
    :param feat: output from lstm
    :param batch_sent_length: (BatchSize, 1:NumOfWords)
        sentences length in batch
    '''
    # calculate in log domain
    # feats is len(sentence) * tagset_size
    # initialize alpha with a Tensor with values all equal to -10000.

    # Do the forward algorithm to compute the partition function
    batch_size = feats.size(0)
    sent_len = feats.size(1)
    combined_score = self.calculate_all_scores(feats)
    init_alphas = Variable(torch.zeros(
        batch_size, sent_len, self.tagset_size).cuda())

    # START_TAG has all of the score.
    init_alphas[:, 0, :] = combined_score[:, 0, self.tag_to_ix[START_TAG], :]

    for word_idx in range(1, sent_len):
        ## batch_size, self.tagset_size, self.tagset_size
        before_log_sum_exp = init_alphas[:, word_idx-1, :].view(batch_size, self.tagset_size, 1) \
            .expand(batch_size, self.tagset_size, self.tagset_size) \
            + combined_score[:, word_idx, :, :]
        init_alphas[:, word_idx, :] = log_sum_exp(before_log_sum_exp)

    last_alpha = torch.gather(init_alphas, 1, batch_sent_length.view(batch_size, 1, 1).expand(
        batch_size, 1, self.tagset_size)-1).view(batch_size, self.tagset_size)
    last_alpha += self.transitions[:, self.tag_to_ix[STOP_TAG]].view(
        1, self.tagset_size).expand(batch_size, self.tagset_size)
    last_alpha = log_sum_exp(last_alpha.view(
        batch_size, self.tagset_size, 1)).view(batch_size)

    # Z(x)
    return torch.sum(last_alpha)


def forward_labeled(self, all_scores: Tensor, word_seq_lens: Tensor, tags: Tensor) -> Tensor:
    '''
    Calculate the scores for the gold instances.
    :param all_scores: (batch, seq_len, tagset_size, tagset_size)
    :param word_seq_lens: (BatchSize, 1:NumOfWords)
        sentences length in batch
    :param tags: (batch, seq_len)
    :param masks: batch, seq_len
    :return: sum of score for the gold sequences Shape: (batch_size)
    '''
    batchSize = all_scores.shape[0]
    sentLength = all_scores.shape[1]
    masks = tags.ne(0.)
    # all the scores to current labels: batch, seq_len, all_from_label?
    currentTagScores = torch.gather(all_scores, 3, tags.view(batchSize, sentLength, 1, 1).expand(
        batchSize, sentLength, self.tagset_size, 1)).view(batchSize, -1, self.tagset_size)
    if sentLength != 1:
        tagTransScoresMiddle = torch.gather(
            currentTagScores[:, 1:, :], 2, tags[:, : sentLength - 1].view(batchSize, sentLength - 1, 1)).view(batchSize, -1)
    tagTransScoresBegin = currentTagScores[:, 0, self.tag_to_ix[START_TAG]]
    endTagIds = torch.gather(tags, 1, word_seq_lens.view(batchSize, 1) - 1)
    tagTransScoresEnd = torch.gather(self.transitions[:, self.tag_to_ix[STOP_TAG]].view(
        1, self.tagset_size).expand(batchSize, self.tagset_size), 1,  endTagIds).view(batchSize)
    score = torch.sum(tagTransScoresBegin) + torch.sum(tagTransScoresEnd)
    if sentLength != 1:
        score += torch.sum(tagTransScoresMiddle.masked_select(masks[:, 1:]))
    return score


def calculate_all_scores(self, feats: Tensor):
    """
    Calculate all scores by adding up the transition scores and emissions (from lstm).
    Basically, compute the scores for each edges between labels at adjacent positions.
    This score is later be used for forward-backward inference
    :param feats: emission scores from lstm.
    :return:
    """
    batch_size = feats.size(0)
    sent_len = feats.size(1)

    # combine score here, feats+ transitions
    combined_score = self.transitions.view(1, 1, self.tagset_size, self.tagset_size) \
                         .expand(batch_size, sent_len, self.tagset_size, self.tagset_size) +\
        feats.view(batch_size, sent_len, 1, self.tagset_size) \
        .expand(batch_size, sent_len, self.tagset_size, self.tagset_size)
    return combined_score

#######################
### Viterbi Decode  ###
#######################


def viterbi_decode(self, all_scores: Tensor, word_seq_lens: Tensor):
    """
    Use viterbi to decode the instances given the scores and transition parameters
    :param all_scores: (batch_size x max_seq_len x num_labels)
    :param word_seq_lens: (batch_size)
    :return: the best scores as well as the predicted label ids.
            (batch_size) and (batch_size x max_seq_len)
    """
    batchSize = all_scores.shape[0]
    sentLength = all_scores.shape[1]
    # sent_len =
    scoresRecord = torch.zeros(
        [batchSize, sentLength, self.tagset_size]).to(self.device)
    idxRecord = torch.zeros(
        [batchSize, sentLength, self.tagset_size], dtype=torch.int64).to(self.device)
    mask = torch.ones_like(word_seq_lens, dtype=torch.int64).to(self.device)
    startIds = torch.full((batchSize, self.tagset_size),
                          self.tag_to_ix[START_TAG], dtype=torch.int64).to(self.device)
    decodeIdx = torch.LongTensor(batchSize, sentLength).to(self.device)

    scores = all_scores
    # scoresRecord[:, 0, :] = self.getInitAlphaWithBatchSize(batchSize).view(batchSize, self.tagset_size)
    # represent the best current score from the start, is the best
    scoresRecord[:, 0, :] = scores[:, 0, self.tag_to_ix[START_TAG], :]
    idxRecord[:,  0, :] = startIds
    for wordIdx in range(1, sentLength):
        # scoresIdx: batch x from_label x to_label at current index.
        scoresIdx = scoresRecord[:, wordIdx - 1, :].view(batchSize, self.tagset_size, 1).expand(batchSize, self.tagset_size,
                                                                                                self.tagset_size) + scores[:, wordIdx, :, :]
        # the best previous label idx to crrent labels
        idxRecord[:, wordIdx, :] = torch.argmax(scoresIdx, 1)
        scoresRecord[:, wordIdx, :] = torch.gather(scoresIdx, 1, idxRecord[:, wordIdx, :].view(
            batchSize, 1, self.tagset_size)).view(batchSize, self.tagset_size)

    lastScores = torch.gather(scoresRecord, 1, word_seq_lens.view(batchSize, 1, 1).expand(
        batchSize, 1, self.tagset_size) - 1).view(batchSize, self.tagset_size)  # select position
    lastScores += self.transitions[:, self.tag_to_ix[STOP_TAG]
                                   ].view(1, self.tagset_size).expand(batchSize, self.tagset_size)
    decodeIdx[:, 0] = torch.argmax(lastScores, 1)
    bestScores = torch.gather(
        lastScores, 1, decodeIdx[:, 0].view(batchSize, 1))

    for distance2Last in range(sentLength - 1):
        lastNIdxRecord = torch.gather(idxRecord, 1, torch.where(word_seq_lens - distance2Last - 1 > 0, word_seq_lens - distance2Last - 1,
                                                                mask).view(batchSize, 1, 1).expand(batchSize, 1, self.tagset_size)).view(batchSize, self.tagset_size)
        decodeIdx[:, distance2Last + 1] = torch.gather(
            lastNIdxRecord, 1, decodeIdx[:, distance2Last].view(batchSize, 1)).view(batchSize)

    return bestScores, decodeIdx

"""
def viterbi_algo(self, feats):
    '''
    In this function, we implement the viterbi algorithm explained above.
    A Dynamic programming based approach to find the best tag sequence
    '''
    backpointers = []
    # analogous to forward

    # Initialize the viterbi variables in log space
    init_vvars = Tensor(1, self.tagset_size).fill_(-10000.)
    init_vvars[0][self.tag_to_ix[START_TAG]] = 0

    # forward_var at step i holds the viterbi variables for step i-1
    forward_var = Variable(init_vvars)
    if self.use_gpu:
        forward_var = forward_var.cuda()
    for feat in feats:
        next_tag_var = forward_var.view(
            1, -1).expand(self.tagset_size, self.tagset_size) + self.transitions
        _, bptrs_t = torch.max(next_tag_var, dim=1)
        # holds the backpointers for this step
        bptrs_t = bptrs_t.squeeze().data.cpu().numpy()
        next_tag_var = next_tag_var.data.cpu().numpy()
        # holds the viterbi variables for this step
        viterbivars_t = next_tag_var[range(len(bptrs_t)), bptrs_t]
        viterbivars_t = Variable(torch.FloatTensor(viterbivars_t))
        if self.use_gpu:
            viterbivars_t = viterbivars_t.cuda()

        # Now add in the emission scores, and assign forward_var to the set
        # of viterbi variables we just computed
        forward_var = viterbivars_t + feat
        backpointers.append(bptrs_t)

    # Transition to STOP_TAG
    terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
    terminal_var.data[self.tag_to_ix[STOP_TAG]] = -10000.
    terminal_var.data[self.tag_to_ix[START_TAG]] = -10000.
    best_tag_id = argmax(terminal_var.unsqueeze(0))
    path_score = terminal_var[best_tag_id]

    # Follow the back pointers to decode the best path.
    best_path = [best_tag_id]
    for bptrs_t in reversed(backpointers):
        best_tag_id = bptrs_t[best_tag_id]
        best_path.append(best_tag_id)

    # Pop off the start tag (we dont want to return that to the caller)
    start = best_path.pop()
    assert start == self.tag_to_ix[START_TAG]  # Sanity check
    best_path.reverse()
    return path_score, best_path


def forward_calc(self, sentence: Tensor, chars2: Tensor, batch_sent_char_length: Tensor, batch_sent_length: Tensor, maxchars: int):
    '''
    The function calls viterbi decode and generates the 
    most probable sequence of tags for the sentence
    '''

    # Get the emission scores from the BiLSTM
    feats = self._get_lstm_features(
        sentence, chars2, batch_sent_char_length, batch_sent_length, maxchars)
    # viterbi to get tag_seq

    # Find the best path, given the features.
    if self.use_crf:
        score, tag_seq = self.viterbi_decode(feats)
    else:
        score, tag_seq = torch.max(feats, 1)
        tag_seq = list(tag_seq.cpu().data)

    return score, tag_seq
"""

def forward_(self, sentence: Tensor, chars2: Tensor, batch_sent_char_length: Tensor, batch_sent_length: Tensor, maxchars: int):
    """
    Calculate the negative log-likelihood
    :param sentence: 2d tensor of size (BatchSize, MaxWordsNum) of sentences in batch
        to be passed to word embedding layer to become 3d tensor 
        (BatchSize, MaxWordsNum, EmbeddingDim)
    :param chars2: 3d tensor of size (BatchSize, MaxNumOfWords, MaxNumOfChars)
    :param batch_sent_char_length: (BatchSize, MaxNumofWords:NumofChars) 
        to keep the words length info for sentences in batch
    :param batch_sent_length: (BatchSize, 1:NumOfWords)
        sentences length in batch
    :param maxchars: max number of chars of words in batch
    :return: NLL score, predicted tag sequence
    """
    feats = self._get_lstm_features(
        sentence, chars2, batch_sent_char_length, batch_sent_length, maxchars)
    all_scores = self.calculate_all_scores(feats)
    score, tag_seq = self.viterbi_decode(all_scores, batch_sent_length)

    return score, tag_seq

#######################
### LSTM features   ###
#######################


def get_lstm_features(self, sentence: Tensor, chars2: Tensor, batch_sent_char_length: Tensor, batch_sent_length: Tensor, maxchars: int=0):
    """
    Complete the forward(in both direction) calculation
    :param sentence: 2d tensor of size (BatchSize, MaxWordsNum) of sentences in batch
        to be passed to word embedding layer to become 3d tensor 
        (BatchSize, MaxWordsNum, EmbeddingDim)
    :param chars2: 3d tensor of size (BatchSize, MaxNumOfWords, MaxNumOfChars)
    :param batch_sent_char_length: (BatchSize, MaxNumofWords:NumofChars) 
        to keep the words length info for sentences in batch
    :param batch_sent_length: (BatchSize, 1:NumOfWords)
        sentences length in batch
    :param maxchars: max number of chars of words in batch
    """
    if self.char_mode == 'LSTM':
        sent_len = sentence.size(1)
        batch_size = sentence.size(0)
        chars2 = chars2.view(batch_size*sent_len, -1)
        batch_sent_char_length = batch_sent_char_length.view(
            batch_size*sent_len)
        chars_embeds = self.char_embeds(chars2)  # .transpose(1, 2)
        chars_embeds = self.dropout(chars_embeds)
        if batch_size == 1:
            outputs, _ = self.char_lstm(chars_embeds)
            maxchars += 1
        else:
            packed = pack_padded_sequence(
                chars_embeds, batch_sent_char_length, batch_first=True, enforce_sorted=False)

            char_lstm_out, _ = self.char_lstm(packed)

            outputs, output_lengths = pad_packed_sequence(
                char_lstm_out, batch_first=True)

        # maxpool along the 2nd dim (chars) as Wang et. al
        # Alternative to maxpool on this part can be possible improvement to the model
        outputs = outputs.view(batch_size, sent_len,
                               maxchars, 2*self.char_embedding_dim)
        pooled = torch.max(outputs, dim=2, keepdim=True).values
        pooled = pooled.squeeze(dim=2)
        pooled = pooled.transpose(1, 0).contiguous().view(
            batch_size, sent_len, -1)
        chars_embeds = pooled
    if self.char_mode == 'CNN':
        chars_embeds = self.char_embeds(chars2).unsqueeze(1)

        # Creating Character level representation using Convolutional Neural Netowrk
        # followed by a Maxpooling Layer
        chars_cnn_out3 = self.char_cnn3(chars_embeds)
        chars_embeds = nn.functional.max_pool2d(
            chars_cnn_out3, kernel_size=(chars_cnn_out3.size(2), 1)).view(chars_cnn_out3.size(0), self.out_channels)

    # Loading word embeddings
    embeds_en = self.word_embeds_en(sentence)
    embeds_es = self.word_embeds_es(sentence)

    # We concatenate the word embeddings and the character level representation
    # to create unified representation for each word
    # Concatenate on 3rd dim (1st dim being batch, 2nd dim being sent length)
    embeds = torch.cat((embeds_en, embeds_es, chars_embeds), 2)

    #embeds = embeds.unsqueeze(1)

    # Dropout on the unified embeddings
    embeds = self.dropout(embeds)

    # Word lstm
    # Takes words as input and generates a output at each step

    if self.word_mode == "LSTM":
        packed_words = pack_padded_sequence(
            embeds, batch_sent_length, batch_first=True, enforce_sorted=False)

        lstm_out, _ = self.lstm(packed_words)
        lstm_out = pad_packed_sequence(lstm_out, batch_first=True)[0]
    elif self.word_mode == "CNN":
        if parameters['cnn_layers'] == 3:
            cnn_out = self.word_cnn1(embeds.unsqueeze(1))
            cnn_out = self.word_cnn2(cnn_out)
            cnn_out = self.word_cnn3(cnn_out)
        elif parameters['cnn_layers'] == 1:
            cnn_out = self.word_cnn(embeds.unsqueeze(1))

        if self.dilation == 0:
            pool_out = nn.functional.max_pool2d(
                cnn_out, kernel_size=(1, cnn_out.size(3)))
            lstm_out = pool_out
        else:
            lstm_out = cnn_out

    # Reshaping the outputs from the lstm layer
    #lstm_out = lstm_out.view(sent_len, self.hidden_dim*2)

    # Dropout on the lstm output
    lstm_out = self.dropout(lstm_out)

    # Linear layer converts the ouput vectors to tag space
    lstm_feats = self.hidden2tag(lstm_out)

    if self.use_crf is False:
        lstm_feats = nn.functional.log_softmax(lstm_feats)
    return lstm_feats


def get_neg_log_likelihood(self, sentence, tags, chars2, batch_sent_char_length, batch_sent_length, maxchars):
    """
    Performs the backward calc
    Returns the predNLL - GoldNLL
    """
    # sentence, tags is a list of ints
    # features is a 2D tensor, len(sentence) * self.tagset_size
    feats = self._get_lstm_features(
        sentence, chars2, batch_sent_char_length, batch_sent_length, maxchars)

    if self.use_crf:
        forward_score = self._forward_alg(feats, batch_sent_length)
        gold_score = self.forward_labeled(
            self.calculate_all_scores(feats), batch_sent_length, tags)
        return forward_score - gold_score
    else:
        tags = Variable(tags)
        scores = nn.functional.cross_entropy(feats, tags)
        return scores

###########################
###########################
####                    ###
### LSTM_CRF Main Class ###
###                     ###
###########################
###########################


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim,
                 char_to_ix=None, en_word_embeds=None, es_word_embeds=None, char_out_dimension=32, char_embedding_dim=32,
                 char_lstm_dim=32, use_gpu=False, use_crf=True, char_mode='CNN', word_mode='LSTM', dilation=False):
        '''
        Input parameters:

        :param vocab_size  Size of vocabulary (int)
        :param tag_to_ix  Dictionary that maps NER tags to indices
        :param embedding_dim  Dimension of word embeddings (int)
        :param hidden_dim  The hidden dimension of the LSTM layer (int)
        :param char_to_ix  Dictionary that maps characters to indices
        :param en_word_embeds = Numpy array which provides mapping from word embeddings to word indices
        :param es_word_embeds
        :param char_out_dimension  Output dimension from the CNN encoder for character
        :param char_embedding_dim  Dimension of the character embeddings
        :param use_gpu defines availability of GPU, 
            when True: CUDA function calls are made
            else: Normal CPU function calls are made
        :param use_crf parameter which decides if you want to use the CRF layer for output decoding
        '''

        super(BiLSTM_CRF, self).__init__()

        # parameter initialization for the model
        self.use_gpu = use_gpu
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.use_crf = use_crf
        self.tagset_size = len(tag_to_ix)
        self.out_channels = char_out_dimension
        self.char_mode = char_mode
        self.word_mode = word_mode
        self.char_embedding_dim = char_embedding_dim
        self.char_lstm_dim = char_lstm_dim
        self.dilation = 1 if dilation is True else 0
        self.tag_to_ix = tag_to_ix
        if use_gpu:
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        if char_embedding_dim is not None:
            self.char_embedding_dim = char_embedding_dim

            # Initializing the character embedding layer
            self.char_embeds = nn.Embedding(
                len(char_to_ix), char_embedding_dim)
            init_embedding(self.char_embeds.weight)

            # Performing LSTM encoding on the character embeddings
            if self.char_mode == 'LSTM':
                self.char_lstm = nn.LSTM(
                    char_embedding_dim, char_lstm_dim, num_layers=1, bidirectional=True, batch_first=True)
                init_lstm(self.char_lstm)

            # Performing CNN encoding on the character embeddings
            if self.char_mode == 'CNN':
                self.char_cnn3 = nn.Conv2d(in_channels=1, out_channels=self.out_channels,
                                           kernel_size=(3, char_embedding_dim), padding=(2, 0))

        # Creating Embedding layer with dimension of ( number of words * dimension of each word)
        # Need to create two for both Eng and Esp
        self.word_embeds_en = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeds_es = nn.Embedding(vocab_size, embedding_dim)
        if en_word_embeds is not None and es_word_embeds is not None:
            # Initializes the word embeddings with pretrained word embeddings
            self.pre_word_embeds = True
            self.word_embeds_en.weight = nn.Parameter(
                torch.FloatTensor(en_word_embeds))
            self.word_embeds_es.weight = nn.Parameter(
                torch.FloatTensor(es_word_embeds))
        else:
            self.pre_word_embeds = False

        # Initializing the dropout layer, with dropout specificed in parameters
        self.dropout = nn.Dropout(parameters['dropout'])

        # Lstm Layer:
        # input dimension: word embedding dimension + character level representation
        # bidirectional=True, specifies that we are using the bidirectional LSTM
        if self.char_mode == 'LSTM':
            self.lstm = nn.LSTM(embedding_dim*2+char_lstm_dim*2,
                                hidden_dim, bidirectional=True, batch_first=True)
        if self.char_mode == 'CNN':
            self.lstm = nn.LSTM(
                embedding_dim*2+self.out_channels, hidden_dim, bidirectional=True)
        # Dilated kernel size: [42, 29, 10] Undilated kernel size: [42,42,42]
        kernel_size = [42, 29, 10] if dilation is True else [
            20, 45, 61]  # [20, 45, 61] with max pooling
        if self.word_mode == 'CNN':
            if parameters['cnn_layers'] == 3:
                self.word_cnn1 = nn.Conv2d(in_channels=1, out_channels=int(hidden_dim/2),
                                           kernel_size=(1, kernel_size[0]), stride=1, dilation=pow(1, self.dilation))
                self.word_cnn2 = nn.Conv2d(in_channels=int(hidden_dim/2), out_channels=hidden_dim,
                                           kernel_size=(1, kernel_size[1]), stride=1, dilation=pow(2, self.dilation))
                self.word_cnn3 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim*2,
                                           kernel_size=(1, kernel_size[2]), stride=1, dilation=pow(3, self.dilation))
            elif parameters['cnn_layers'] == 1:
                self.word_cnn = nn.Conv2d(in_channels=1, out_channels=hidden_dim*2,
                                          kernel_size=(1, 100))
            else:
                raise "currently only support 1 or 3 layers"
        # Initializing the lstm layer using predefined function for initialization
        init_lstm(self.lstm)
        # Linear layer which maps the output of the bidirectional LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim*2, self.tagset_size)

        # Initializing the linear layer using predefined function for initialization
        init_linear(self.hidden2tag)

        if self.use_crf:
            # Matrix of transition parameters.  Entry i,j is the score of transitioning *to* i *from* j.
            # Matrix has a dimension of (total number of tags * total number of tags)
            self.transitions = nn.Parameter(
                torch.zeros(self.tagset_size, self.tagset_size))

            # These two statements enforce the constraint that we never transfer
            # to the start tag and we never transfer from the stop tag
            self.transitions.data[:, tag_to_ix[START_TAG]] = -10000
            self.transitions.data[tag_to_ix[STOP_TAG], :] = -10000
            self.transitions.data[tag_to_ix[PAD_TAG], :] = -10000
            self.transitions.data[:, tag_to_ix[PAD_TAG]] = -10000

    # assigning the functions, which we have defined earlier
    _score_sentence = score_sentences
    _get_lstm_features = get_lstm_features
    _forward_alg = forward_pred
    viterbi_decode = viterbi_decode
    neg_log_likelihood = get_neg_log_likelihood
    forward = forward_
    calculate_all_scores = calculate_all_scores
    forward_labeled = forward_labeled
