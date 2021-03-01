'''
--- Neural Networks prject ---
Topic: Sequence-to-sequence in Neural Machine Translation
Authors: Nicholas Redi & Eleonora Vitanza
'''

import os
import sys
import argparse
import random
import torch
from torchtext.data.metrics import bleu_score
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# PREFIX_COLAB is just a prefix string that will be used to run experiments on Google Colab
# it will remain empty for experiments running in a local computer
PREFIX_COLAB = ""
# the PRETRAINED_EMBEDDINGS structure will change to [True, '<embed_file>'] when we want to use pretrained embeddings
PRETRAINED_EMBEDDINGS = [False, '']

# these are the global hyperparameters that will be setted by the command line parser
LR = None
EMBEDDING_DIM = None
HIDDEN_DIM = None
DROPOUT = None
N_LAYERS = None
BIDIRECTIONAL = None
ATTENTION = None
TEACHER_FORCING_RATIO = None
EPOCHS = None
BATCH_SIZE = None
FRAC = None

def build_dataset_list(file_name):
    '''
    Args: the name of the dataset file (*.txt)
    Returns: a list of lists, e.g. [['Hello.','Ciao.'],['The pen is on the table.','La penna è sul tavolo.'],...]
    '''

    if not os.path.isfile(file_name):
        raise Exception(f"Sorry, `{file_name}´ does not exist.")
    if not file_name.endswith('.txt'):
        raise Exception("Sorry, the file must a text file (*.txt).")

    dataset_list = []
    with open(file_name, 'r') as f:
        for line in f.readlines():
            dataset_list.append(line.split('\t')[0:2]) # The dataset contains other irrelevant features, we take only the pair of sequences

    return dataset_list


def tokenize(token):
    '''
    We use this function to handle the creation of the vocabularies.
    Args: a token that could have some symbols inside, e.g. `Hello,´
    Returns: the right tokenization, e.g. `Hello´ , `,´
    '''

    # we suppose that each token could be splitten in three parts
    token2 = None
    token3 = None

    # the case in which the token is just `...´
    if token =='...':
        return token, token2, token3

    # the case in which the token is something like `hello...´ or `...so´
    if '...' in token:
        if '...' in (token[0:3], token[-3:0]):
            token = token.strip('...')
            token2 = '...'
        else:
            token , token2 = token.split('...')
            token3 = '...'
        return token, token2, token3

    # the case in which the token has some symbols attached or it is just a symbol
    symbols = [',',';','.',':','-','_','"','?','!','<','>','=','(',')','[',']','$','/','%','*','+','@']
    for elem in symbols:

        if elem == token:
            return token, token2, token3

        if elem in token:
            if elem in (token[0], token[-1]):
                token2 = elem
                token = token.strip(elem)
                if token[-1] in symbols:
                    token3 = token[-1]
                    token = token.strip(token[-1])

    return token, token2, token3


def create_vocabulary(sequences_list):
    '''
    Args: a list of sentences in the input or target language
    Returns: a dictionary representing the vocabulary of that language { 'Word' : [word_id, frequency] , ... }
    '''

    # build the vocabulary with the frequency of each word
    vocab = {}

    for sequence in sequences_list:
        tokens_list = sequence.split()

        for token in tokens_list:
            token, token2, token3 = tokenize(token)

            if token not in vocab:
                vocab[token] = 1
            else:
                vocab[token] += 1

            if token2 is not None:
                if token2 not in vocab:
                    vocab[token2] = 1
                else:
                    vocab[token2] += 1

            if token3 is not None:
                if token3 not in vocab:
                    vocab[token3] = 1
                else:
                    vocab[token3] += 1

    # taking the most frequent words (minimum frequence of 4) and ordering the vocabulary by the frequency
    cutted_vocab = [[key, vocab[key]] for key in vocab if vocab[key] > 3]
    vocab = {}
    i = 0
    for item in cutted_vocab:
        vocab[item[0]] = [i , item[1]]
        i +=1
    vocab = dict(sorted(vocab.items(), key=lambda item: item[1][1], reverse=True))

    # adding special words
    vocab['<sos>'] = [i, 'irrelevant'] # start-of-sequence
    vocab['<eos>'] = [i + 1, 'irrelevant'] # end-of-sequence
    vocab['<pad>'] = [i + 2, 'irrelevant'] # padding symbol
    vocab['<unk>'] = [i + 3, 'irrelevant'] # unknown symbol (out-of-vocabulary)

    return vocab


def guess_embed_size(embed_file):
    '''
    Just read the first line of an embedding file to guess the size.
    '''

    with open(embed_file, 'r') as f:
        line = f.readline()
        return len(line.split()[1:])


def build_embedding_tensor(vocab, embed_file):
    '''
    Build the embedding tensor to use in case of pretrained embeddings.
    '''

    # initalize the tensor
    embedding_tensor = torch.randn((len(vocab), EMBEDDING_DIM), dtype=torch.float32)
    # we assume the embed_file to be structured like: word feature_1 ... feature_N
    with open(embed_file,"r") as f:
        for line in f.readlines():
            line = line.split()
            word = line[0]
            embedd_values = [float(elem) for elem in line[1:]]
            if word in vocab:
                # we insert the embedding values at i-th row, where i is the id of the word in the vocabulary
                embedding_tensor[vocab[word][0],:] = torch.tensor(embedd_values, dtype=torch.float32)

    return embedding_tensor


def create_minibatch(batch_list, input_vocab, target_vocab):
    '''
    The function pack a batch of examples into a tensor to feed the model.
    Args:
    - batch_list: a list of lists from dataset
    - input_vocab: the input language vocabulary
    - target_vocab: the target language vocabulary
    Returns:
    - input_batch_tensor: the tensor that will feed the Encoder
        -> each element is the id representing one word in one sentence of the input batch
    - target_batch_tensor: the tensor that will feed the Decoder
        -> each element is the id representing one word in one sentence of the target batch
    - input_batch_len: the tensor with the lenghts (no padding) of the sentences in input_batch_tensor (used for `pack_padded_sequence´)
    '''

    # separate the batch list by column
    batch_list_input = [row[0] for row in batch_list]
    batch_list_target = [row[1] for row in batch_list]

    # sorting both columns according to the lenght of the input sequences
    input_batch_list = sorted(batch_list_input, key = lambda x: len(tokenize_sequence(x)), reverse=True)
    target_batch_list = [batch_list_target[batch_list_input.index(elem)] for elem in input_batch_list]

    # taking the input max sequence length
    input_max_sequence_length = len(tokenize_sequence(input_batch_list[0]))
    # the input_batch_tensor will have dimension: lenght of the longest sequence in the batch + 2 , lenght of the batch
    # each sequence is eventually padded with <pad> symbol and have <sos> and <eos> symbols at the extremes
    input_batch_tensor = torch.zeros((input_max_sequence_length + 2, len(input_batch_list)), dtype=torch.int64)
    # the input_batch_len will have dimension: len(input_batch_list)
    input_batch_len = torch.zeros(len(input_batch_list), dtype=torch.int64)
    # filling the tensors
    for col, sequence in enumerate(input_batch_list):
        input_batch_len[col] = len(ids_sequence(tokenize_sequence(sequence), input_vocab, input_max_sequence_length,with_pad=False))
        # in case of attention mechanism or bidirectional lstm we do not revert the input sequence
        if ATTENTION or BIDIRECTIONAL:
            sentence = ids_sequence(tokenize_sequence(sequence), input_vocab, input_max_sequence_length)
        else:
            sentence = ids_sequence(tokenize_sequence(sequence)[::-1], input_vocab, input_max_sequence_length)
        input_batch_tensor[:, col] = torch.tensor(sentence, dtype=torch.int32)

    # we use target_batch_list_temp just to find the target max sequence lenght
    target_batch_list_temp = sorted(target_batch_list, key = lambda x: len(tokenize_sequence(x)), reverse=True)
    target_max_sequence_length = len(tokenize_sequence(target_batch_list_temp[0]))
    # filling the target batch tensor
    target_batch_tensor = torch.zeros((target_max_sequence_length+2, len(target_batch_list)), dtype=torch.int64)
    for col, sequence in enumerate(target_batch_list):
        sentence = ids_sequence(tokenize_sequence(sequence), target_vocab, target_max_sequence_length)
        target_batch_tensor[:, col] = torch.tensor(sentence, dtype=torch.int32)

    return input_batch_tensor, target_batch_tensor, input_batch_len


def get_word(index, vocab):
    '''
    From the index we reach the word in the vocabulary.
    '''

    for word, value in vocab.items():
        if value[0] == index:
            return word

    return None


def create_splits(dataset_list, train_percentage, val_percentage):
    '''
    We create train, validation and test set, saving each of them in a file.
    Args:
    - dataset_list: a list of lists of the dataset
    - train_percentage: the piece of data used to train the model
    - val_percentage: the piece of data used to validate the model
    Returns:
    - train_set: a list of lists representing the train set
    - val_set: a list of lists representing the validation set
    '''

    if not train_percentage + val_percentage < 1:
        raise ValueError('(train_percentage + val_percentage) must be less than 1')

    # initialize a shuffled list of indices
    indices = list(range(0,len(dataset_list)))
    random.shuffle(indices)

    # selecting the size of the splits
    train_len = round(train_percentage * len(dataset_list))
    val_len = round(val_percentage * len(dataset_list))

    # creating subsets of shuffled indices
    shuffled_indices_train = indices[0:train_len]
    shuffled_indices_val = indices[train_len: train_len + val_len]
    shuffled_indices_test = indices[train_len + val_len:]

    # saving the train set file...
    train_set = [dataset_list[i] for i in shuffled_indices_train]
    with open(PREFIX_COLAB + f"dataset/splits_frac_{FRAC}/train_set.txt", "w") as f:
        for elem in train_set:
            f.write(f"{elem[0]}\t{elem[1]}\n")

    # saving the validation set file...
    val_set = [dataset_list[i] for i in shuffled_indices_val]
    with open(PREFIX_COLAB + f"dataset/splits_frac_{FRAC}/val_set.txt", "w") as f:
        for elem in val_set:
            f.write(f"{elem[0]}\t{elem[1]}\n")

    # saving the test set file...
    test_set = [dataset_list[i] for i in shuffled_indices_test]
    with open(PREFIX_COLAB + f"dataset/splits_frac_{FRAC}/test_set.txt", "w") as f:
        for elem in test_set:
            f.write(f"{elem[0]}\t{elem[1]}\n")

    # since we use this function in train mode only, we just need train set and validation set as outcomes
    return train_set, val_set


def load_dataset_splits():
    '''
    This function loads splitten datasets from files (created by `create_splits´ function).
    We assume to have those files in a fixed path, hence no file name as input is needed.
    '''

    # loading train set...
    train_set = []
    with open(PREFIX_COLAB + f"dataset/splits_frac_{FRAC}/train_set.txt", 'r') as f:
        for line in f.readlines():
            train_set.append(line.split('\t'))

    # loading validation set...
    val_set = []
    with open(PREFIX_COLAB + f"dataset/splits_frac_{FRAC}/val_set.txt", 'r') as f:
        for line in f.readlines():
            val_set.append(line.split('\t'))

    # loading test set...
    test_set = []
    with open(PREFIX_COLAB + f"dataset/splits_frac_{FRAC}/test_set.txt", 'r') as f:
        for line in f.readlines():
            test_set.append(line.split('\t'))

    return train_set, val_set, test_set


def tokenize_sequence(sequence):
    '''
    The function tokenizes a given sequence using a similar approach of the `tokenize´ function.
    It returns a list of tokens.
    '''

    # first, we split the sequence by spaces
    sequence = sequence.split()

    # initialize the tokens' list
    tokenized_sequence = []
    for token in sequence:

        # we handle the case of `...´
        if '...' in token:
            if token[0:3] == '...':
                tokenized_sequence.append('...')
                tokenized_sequence.append(token.strip('...'))
            elif token[-3:0] == '...':
                tokenized_sequence.append(token.strip('...'))
                tokenized_sequence.append('...')
            else:
                wl, wr = token.split('...')
                tokenized_sequence.append(wl)
                tokenized_sequence.append('...')
                tokenized_sequence.append(wr)
            continue

        # we handle the other symbols
        symbols = [',',';','.',':','-','_','"','?','!','<','>','=','(',')','[',']','$','/','%','*','+','@']
        # we use a flag to manage tokens without any symbol
        there_is_a_symbol = False
        for elem in symbols:

            # if the token is just one symbol...
            if elem == token:
                tokenized_sequence.append(elem)
                there_is_a_symbol = True
                break

            # if the token contains a symbol...
            # We assume at most one symbol for each token, eventually repeated at the beginning and at the end
            # of the token
            if elem in token:
                # symbol at the beginning
                if elem == token[0]:
                    tokenized_sequence.append(elem)
                    tokenized_sequence.append(token[1:])
                # symbol at the end
                elif elem == token[-1]:
                    tokenized_sequence.append(token[0:-1])
                    tokenized_sequence.append(elem)
                # same symbol at both the beginning and the end
                elif elem == token[0] and elem == token[-1]:
                    tokenized_sequence.append(elem)
                    tokenized_sequence.append(token.strip(elem))
                    tokenized_sequence.append(elem)
                # symbol inside the token
                else:
                    # it could be the case of: `It's´ or `trade-off´
                    tokenized_sequence.append(token)

                there_is_a_symbol = True
                break

        # if the token does not contain any symbol...
        if not there_is_a_symbol:
            tokenized_sequence.append(token)

    return tokenized_sequence


def ids_sequence(tokenized_sequence, vocab, max_sequence_length, with_pad=True):
    '''
    We used this function to convert tokens into ids.
    Args:
    - tokenized_sequence: an already tokenized sentence
    - vocab: the referred vocabulary
    - max_sequence_length: the lenght of the longest sentence in the batch
    - with_pad: boolean to add or remove padding
    Returns: a list containing the ids of the sentence, the <sos> and <eos> ids, and eventually the padding id.
    '''

    # initialize the list with the <sos> id
    ids_sequence = [vocab['<sos>'][0]]

    # converting tokens into ids
    for token in tokenized_sequence:
        if token in vocab:
            ids_sequence.append(vocab[token][0])
        else:
            # if a token is out of vocabulary...
            ids_sequence.append(vocab['<unk>'][0])

    # append the final <eos> id
    ids_sequence.append(vocab['<eos>'][0])

    if with_pad:
        # fill the gap with padding
        while len(ids_sequence) < max_sequence_length+2:
            ids_sequence.append(vocab['<pad>'][0])

    return ids_sequence


class Encoder(torch.nn.Module):
    '''
    We use a LSTM to map the input sequence to a fixed dimensional vector.
    Specifically, the encoder consists of one Embedding module, one Dropout module and one LSTM module.
    '''

    def __init__(self, input_vocab, embedding_dim, hidden_dim, n_layers, dropout_prob, bidirectional, attention, pretrained_embeddings=None):
        super().__init__()

        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.attention = attention
        self.padding_idx = input_vocab['<pad>'][0]

        # case of no pretrained embedding
        if pretrained_embeddings is None:
            self.embedding = torch.nn.Embedding(len(input_vocab), embedding_dim)
        # case of pretrained embedding
        else:
            self.embedding = torch.nn.Embedding.from_pretrained(pretrained_embeddings, freeze=True, padding_idx=self.padding_idx)

        self.dropout = torch.nn.Dropout(dropout_prob)

        # case of no bidirectional lstm
        if self.bidirectional is False:
            self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout_prob)
        # case of bidirectional lstm
        else:
            self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout_prob, bidirectional=True)

    def forward(self, input_batch, input_batch_len):
        '''
        The Encoder takes one input batch and its sequence lenghts (created by the `create_minibatch´ function)
        and returns outputs, hidden and cell tensors.
        '''

        # input_batch.shape = [max_seq_len,batch_size]
        # input_batch_len.shape = [batch_size]

        # applying embedding and dropout layers
        input_batch = self.dropout(self.embedding(input_batch))

        # input_batch.shape = [max_seq_len,batch_size,embedding_dim]

        # gather the data avoiding to process padding
        input_batch = torch.nn.utils.rnn.pack_padded_sequence(input_batch, input_batch_len.cpu(), batch_first=False)

        # input_batch becomes a `PackedSequence´ object

        # applying lstm layer
        outputs, (hidden, cell) = self.lstm(input_batch)

        # outputs is still a `PackedSequence´ object
        # hidden and cell shape = [n_layers*n_directions,batch_size,hidden_dim]

        # fix the dimensions in case of bidirectional lstm
        if self.bidirectional:
            # hidden[0,:,:] and hidden[1,:,:] are 2D tensor so dim=1 refers to the hidden_dim
            hidden = torch.cat([hidden[0,:,:], hidden[1,:,:]], dim=1).unsqueeze(0)
            cell = torch.cat([cell[0,:,:], cell[1,:,:]], dim=1).unsqueeze(0)
            # hidden and cell shape = [1,batch_size,2*hidden_dim]

        # padding `outputs´ in case of attention mechanism (it becomes a tensor)
        if self.attention:
            outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, padding_value=self.padding_idx)
            # outputs.shape = [max_seq_len,batch_size,hidden_dim]

        return outputs, hidden, cell


class Attention(torch.nn.Module):
    '''
    Global attention mechanism based on two linear layers.
    '''

    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super().__init__()

        self.attn_hidden_vector = torch.nn.Linear(encoder_hidden_dim + decoder_hidden_dim, decoder_hidden_dim)
        self.attn_scoring_fn = torch.nn.Linear(decoder_hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        '''
        We use the hidden representation of the predicted token and the outputs
        of the encoder to compute the attention weights.
        '''

        # hidden.shape = [1,batch_size,decoder_hidden_dim] so we repeat it in dim 0 in order to concatenate
        hidden = hidden.repeat(encoder_outputs.shape[0], 1, 1)
        # hidden.shape = [max_seq_len,batch_size,decoder_hidden_dim]
        # torch.cat((hidden, encoder_outputs), dim=2).shape = [max_seq_len,batch_size,encoder_hidden_dim+decoder_hidden_dim]
        att_hidden = torch.tanh(self.attn_hidden_vector(torch.cat((hidden, encoder_outputs), dim=2)))
        # att_hidden.shape = [max_seq_len,batch_size,decoder_hidden_dim]
        att_scoring_vector = self.attn_scoring_fn(att_hidden).squeeze(2)
        # att_scoring_vector.shape = [max_seq_len,batch_size]
        att_scoring_vector = att_scoring_vector.permute(1, 0)
        # att_scoring_vector.shape = [batch_size,max_seq_len]

        return torch.nn.functional.softmax(att_scoring_vector, dim=1)
        # softmax(att_scoring_vector, dim=1).shape = [batch_size,max_seq_len]


class OneStepDecoder(torch.nn.Module):
    '''
    OneStepDecoder takes the previous predicted word and returns the next one.
    It consists of one Embedding module, one Dropout module, one LSTM module and one Linear module.
    '''

    def __init__(self, target_vocab, embedding_dim, decoder_hidden_dim, n_layers, dropout_prob, encoder_bidirectional, attention=None, encoder_hidden_dim=None):
        super().__init__()

        # we use the one-hot representation at the end of each OneStepDecoder, hence we need the target vocabulary size
        self.target_vocab_size = len(target_vocab)

        self.encoder_bidirectional = encoder_bidirectional
        self.attention = attention

        self.embedding = torch.nn.Embedding(self.target_vocab_size, embedding_dim)

        self.dropout = torch.nn.Dropout(dropout_prob)

        # case of no bidirectional encoder...
        if self.encoder_bidirectional is False:
            # case of attention...
            if self.attention is not None:
                self.lstm = torch.nn.LSTM(encoder_hidden_dim + embedding_dim, decoder_hidden_dim, n_layers, dropout=dropout_prob)
                self.linear = torch.nn.Linear(encoder_hidden_dim + decoder_hidden_dim + embedding_dim, self.target_vocab_size)
            # basic case...
            else:
                self.lstm = torch.nn.LSTM(embedding_dim, decoder_hidden_dim, n_layers, dropout=dropout_prob)
                self.linear = torch.nn.Linear(decoder_hidden_dim, self.target_vocab_size)
        # case of bidirectional encoder...
        else:
            self.lstm = torch.nn.LSTM(embedding_dim, 2 * decoder_hidden_dim, n_layers, dropout=dropout_prob)
            self.linear = torch.nn.Linear(2 * decoder_hidden_dim, self.target_vocab_size)

    def forward(self, target_tokens, hidden, cell, encoder_outputs):
        '''
        OneStepDecoder takes one target token for each sequence of the batch, plus
        the outcomes of the encoder, and returns the one-hot representation
        of the next predicted word.
        '''

        # target_tokens.shape = [batch_size]
        # hidden and cell shape = [n_layers,batch_size,n_directions*hidden_dim]
        # encoder_outputs is a `PackedSequence´ object

        # Since the OneStepDecoder refers to a single word and the embedding
        # layer accepts 2D tensors as input, we add a dummy dimension of size one
        target_tokens = target_tokens.unsqueeze(0)
        # target_tokens.shape = [1,batch_size]
        embedding_layer = self.dropout(self.embedding(target_tokens))
        # embedding_layer.shape = [1,batch_size,embedding_dim]

        # case of attention...
        if self.attention is not None:

            # Calculate the attention weights of shape [max_seq_len,1,batch_size]
            attention_weights = self.attention(hidden, encoder_outputs).unsqueeze(1)
            # attention_weights.shape = [batch_size,1,max_seq_len]

            encoder_outputs = encoder_outputs.permute(1, 0, 2)
            # encoder_outputs.shape = [batch_size,max_seq_len,hidden_dim]

            context_vector = torch.bmm(attention_weights, encoder_outputs)
            # context_vector.shape = [batch_size,1,hidden_dim]
            context_vector = context_vector.permute(1, 0, 2)
            # context_vector.shape = [1,batch_size,hidden_dim]

            # embedding_layer.shape = [1,batch_size,embedding_dim]
            attention_vector = torch.cat((embedding_layer, context_vector), dim=2)
            # attention_vector.shape = [1,batch_size,embedding_dim+hidden_dim]

            output, (hidden, cell) = self.lstm(attention_vector, (hidden, cell))
            # output.shape = [1,batch_size,hidden_dim]
            # hidden and cell shape = [1,batch_size,hidden_dim]

            # we remove the first dimension to apply the Linear module
            # torch.cat((output.squeeze(0), context_vector.squeeze(0), embedding_layer.squeeze(0)), dim=1).shape = [batch_size,hidden_dim+hidden_dim+embedding_dim]
            linear = self.linear(torch.cat((output.squeeze(0), context_vector.squeeze(0), embedding_layer.squeeze(0)), dim=1))
            # linear.shape = [batch_size,target_vocab_size]

            # we remove the second dimension to plot the weights later
            attention_weights = attention_weights.squeeze(1)
            # attention_weights.shape = [batch_size,max_seq_len]

        # case of no attention...
        else:

            output, (hidden, cell) = self.lstm(embedding_layer, (hidden, cell))
            # output.shape : [1,batch_size,n_directions*hidden_dim]
            # hidden and cell shape = [n_layers,batch_size,n_directions*hidden_dim]

            # we remove the first dimension to apply the Linear module
            linear = self.linear(output.squeeze(0))
            # linear.shape = [batch_size,target_vocab_size]

            attention_weights = None # dummy return

        return linear, hidden, cell, attention_weights


class Decoder(torch.nn.Module):
    '''
    We use another LSTM to decode the target sequence from a vector.
    We recursively call the OneStepDecoder.
    '''

    def __init__(self, one_step_decoder):
        super().__init__()

        self.one_step_decoder = one_step_decoder

    def forward(self, target_batch, encoder_outputs, hidden, cell, teacher_forcing_ratio=0):
        '''
        The Decoder takes one target batch (created by the `create_minibatch´ function)
        and returns the predicted sequences of the referred batch.
        '''

        # target_batch.shape = [max_seq_len,batch_size]
        # hidden and cell shape = [n_layers,batch_size,n_directions*hidden_dim]
        # encoder_outputs is a `PackedSequence´ object

        target_len, batch_size = target_batch.shape[0], target_batch.shape[1]
        # initializing the tensor
        predictions = torch.zeros(target_len, batch_size, self.one_step_decoder.target_vocab_size).to(device)
        # predictions.shape = [max_seq_len,batch_size,target_vocab_size]

        # ids of the first words in the target batch
        input_tokens_id = target_batch[0, :]
        # input_tokens_id.shape = [batch_size]

        # loop through the max lenght of the target sequences
        for t in range(1, target_len):

            # at first iteration, `predict´ will contain the one-hot representation of the predicted second words
            # at second iteration, `predict´ will contain the one-hot representation of the predicted third words
            # and so on...
            predict, hidden, cell, _ = self.one_step_decoder(input_tokens_id, hidden, cell, encoder_outputs)
            # predict.shape = [batch_size,target_vocab_size]
            # hidden and cell shape = [n_layers,batch_size,n_directions*hidden_dim]
            predictions[t] = predict

            # picking the ids with max probability
            input_tokens_id = predict.argmax(1)
            # input_tokens_id.shape = [batch_size]

            # in train mode we help the network with the learning process
            # using randomly the target instead of the predicted word
            do_teacher_forcing = random.random() < teacher_forcing_ratio
            input_tokens_id = target_batch[t] if do_teacher_forcing else input_tokens_id

        return predictions


class EncoderDecoder(torch.nn.Module):
    '''
    EncoderDecoder wraps together the Encoder and Decoder classes.
    '''

    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_batch, target_batch, input_batch_len, teacher_forcing_ratio=0):
        '''
        We just call the forward pass of both Encoder and Decoder given input and target batches.
        '''

        encoder_outputs, hidden, cell = self.encoder(input_batch, input_batch_len)
        outputs = self.decoder(target_batch, encoder_outputs, hidden, cell, teacher_forcing_ratio)

        return outputs


def create_model(input_vocab, target_vocab, mode):
    '''
    The function takes the vocabularies and returns the Encoder-Decoder model.
    '''

    ####### ENCODER ##########

    # if no pretrained embedding...
    if not PRETRAINED_EMBEDDINGS[0]:
        encoder = Encoder(input_vocab, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT, BIDIRECTIONAL, ATTENTION)

    # if there is pretrained embedding...
    elif PRETRAINED_EMBEDDINGS[0]:
        input_embed_tensor = build_embedding_tensor(input_vocab, PREFIX_COLAB + "wembedds/" + PRETRAINED_EMBEDDINGS[1])
        encoder = Encoder(input_vocab, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT, BIDIRECTIONAL, ATTENTION, pretrained_embeddings=input_embed_tensor)

    ####### ONE-STEP-DECODER #########

    # if attention mechanism is on...
    if ATTENTION:
        attention_layer = Attention(HIDDEN_DIM, HIDDEN_DIM)
        one_step_decoder = OneStepDecoder(target_vocab, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT, BIDIRECTIONAL, attention_layer, HIDDEN_DIM)

    # if attention mechanism is off...
    else:
        one_step_decoder = OneStepDecoder(target_vocab, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT, BIDIRECTIONAL)

    ####### DECODER #########

    decoder = Decoder(one_step_decoder)

    ####### MODEL #########

    model = EncoderDecoder(encoder, decoder)
    model = model.to(device)

    # prepare the model adding optimizer and criterion for training
    if mode == 'train':

        optimizer = torch.optim.Adam(model.parameters(), LR)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=target_vocab["<pad>"][0])

        return model, optimizer, criterion

    # in 'eval' or 'on_the_fly' mode we just return the model
    if mode in ('eval', 'on_the_fly'):

        return model

    raise Exception('The mode must be "train", "eval" or "on_the_fly".')


def train(train_set, val_set, input_vocab, target_vocab, epochs, batch_size):
    '''
    This function create a model, train it, and returns the best one according to the average validation loss.
    '''

    # create the model
    model, optimizer, criterion = create_model(input_vocab, target_vocab, 'train')

    # saving the best model to return
    best_model = model

    n_train = len(train_set)
    n_val = len(val_set)

    # each element of these lists will be the average loss computed on the batches
    epoch_loss_train = []
    epoch_loss_val = []

    # setting initial values to find the best model
    best_loss_val = 100.
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        print(f"---- epoch {epoch}/{epochs}")

        # each element of this list will be the loss computed on each batch
        batch_loss_train = []

        # setting the model to train mode
        model.train()

        # splitting the train set in minibatches
        t = 0
        nb = 1
        while True:
            f = t
            t = min(f + batch_size, n_train)

            # preparing tensors to feed the model
            input_batch, target_batch, input_batch_len = create_minibatch(train_set[f:t], input_vocab, target_vocab)
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            input_batch_len = input_batch_len.to(device)

            # clear previous computed gradients
            optimizer.zero_grad()

            # forward step
            output = model(input_batch, target_batch, input_batch_len, TEACHER_FORCING_RATIO)

            # compute the loss between output and target
            output = output[1:].view(-1, output.shape[-1])
            target_batch = target_batch[1:].view(-1)
            loss = criterion(output, target_batch)

            # backward step
            loss.backward()

            # normalize to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            # update the weights
            optimizer.step()

            # appending the loss of the current batch
            batch_loss_train.append(loss.item())

            print(f"epoch {epoch} --- batch {nb}/{int(n_train/batch_size)+1} --- train loss = {round(batch_loss_train[-1], 4)}")

            # go to the next batch
            nb += 1
            if t == n_train:
                break

        # adding the average loss
        epoch_loss_train.append(round(sum(batch_loss_train)/len(batch_loss_train), 4))

        # in eval mode we do not need gradients
        with torch.no_grad():

            # each element of this list will be the loss computed on each batch
            batch_loss_val = []

            # setting the model to eval mode
            model.eval()

            # splitting the validation set in minibatches
            t = 0
            nb = 1
            while True:
                f = t
                t = min(f + batch_size, n_val)

                # preparing tensors to feed the model
                input_batch, target_batch, input_batch_len = create_minibatch(val_set[f:t], input_vocab, target_vocab)
                input_batch = input_batch.to(device)
                target_batch = target_batch.to(device)
                input_batch_len = input_batch_len.to(device)

                # forward step with teacher forcing ratio set to zero
                output = model(input_batch, target_batch,input_batch_len, 0)

                # compute the loss between output and target
                output = output[1:].view(-1, output.shape[-1])
                target_batch = target_batch[1:].view(-1)
                loss = criterion(output, target_batch)

                # appending the loss of the current batch
                batch_loss_val.append(loss.item())

                print(f"epoch {epoch} --- batch {nb}/{int(n_val/batch_size)+1} --- validation loss = {round(batch_loss_val[-1], 4)}")

                # go to the next batch
                nb += 1
                if t == n_val:
                    break

        # adding the average loss
        epoch_loss_val.append(round(sum(batch_loss_val)/len(batch_loss_val), 4))

        if epoch_loss_val[-1] < best_loss_val:
            print(f'Found best model at epoch {epoch}')
            best_loss_val = epoch_loss_val[-1]
            best_epoch = epoch
            best_model = model

        print(f"epoch {epoch}: average train loss = {epoch_loss_train[-1]} , average val loss = {epoch_loss_val[-1]}")

    return best_model, best_epoch, epoch_loss_train, epoch_loss_val


def plot(loss_train, loss_val, plot_name):
    '''
    Plot the result of the training routine.
    '''

    _, ax = plt.subplots()
    plt.plot(loss_train, label='Training Data')
    plt.plot(loss_val, label='Validation Data')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.ylim((0, 10))
    plt.legend(loc='lower right')

    validation_loss = min(loss_val)
    epoch = loss_val.index(validation_loss)
    training_loss = loss_train[epoch]
    textstr = f'Best model found at epoch {epoch+1}\nTraining loss: {round(training_loss,2)}\nValidation loss: {round(validation_loss,2)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.35, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    plt.savefig(plot_name)


def test(test_set, model, input_vocab, target_vocab, test_name):
    '''
    Translate and compute BLEU score on the test set.
    '''

    # these lists will contain respectively the target and predicted sequences
    targets = []
    predictions = []

    # create a file to accumulate all the translations
    with open(test_name, 'w') as f:
        sentence_index = 1

        # loop on the test set
        for pair_sequences in test_set:
            print(f'translating sentence {sentence_index} of {len(test_set)}')
            sentence_index += 1
            # prepare the tensor to feed the encoder (we consider it as a batch of size 1)
            input_seq, _, input_batch_len = create_minibatch([pair_sequences], input_vocab, target_vocab)
            input_seq = input_seq.to(device)
            input_batch_len = input_batch_len.to(device)

            # set on evaluation mode
            model.eval()

            # run the forward pass of the encoder
            encoder_outputs, hidden, cell = model.encoder(input_seq, input_batch_len)

            # take the id value of <sos> from the target vocabulary
            sos_index = [target_vocab['<sos>'][0]]
            next_token = torch.LongTensor(sos_index).to(device)

            # this list will contain the predicted words of the current sentence
            predicted_words = []

            # no gradient computations are needed
            with torch.no_grad():
                # In loop run the forward pass of the OneStepDecoder until some specified step (say 30)
                for _ in range(30):
                    output, hidden, cell, _ = model.decoder.one_step_decoder(next_token, hidden, cell, encoder_outputs)

                    # take the most probable word
                    next_token = output.argmax(1)

                    # from id to word
                    predicted_word = get_word(next_token.item(), target_vocab)

                    # stop if the model reaches <eos>
                    if predicted_word == '<eos>':
                        break

                    # appending the predicted word to the list
                    predicted_words.append(predicted_word)

            # writing on the file...
            f.write(f'    input sentence: {pair_sequences[0]}\n')
            f.write(f'   target sentence: {" ".join(tokenize_sequence(pair_sequences[1]))}\n')
            f.write(f'predicted sentence: {" ".join(predicted_words)}\n\n')


            # appending the predicted and target sequences to the lists
            # notice that targets are lists of lists of lists because in the computation of BLEU score
            # you could have more references than one as good translations of the sentence.
            predictions.append(predicted_words)
            targets.append([tokenize_sequence(pair_sequences[1])])

        # compute the BLEU score using targets as reference corpus
        print('Computing BLEU score...')
        bleu = round(bleu_score(predictions, targets) * 100, 2)
        f.write(f'BLEU score: {bleu}')

    print(f'BLEU Score: {bleu}')


def on_the_fly(seq_to_translate, model, input_vocab, target_vocab, att_plot_name):
    '''
    Translate a sentence on the fly.
    '''

    lenght = len(tokenize_sequence(seq_to_translate))
    # we do not revert the input sentence in case of attention or bidirectional
    if ATTENTION or BIDIRECTIONAL:
        ids_seq_to_translate = ids_sequence(tokenize_sequence(seq_to_translate), input_vocab, lenght, with_pad=False)
    # we revert the input sentence in the basic case
    else:
        ids_seq_to_translate = ids_sequence(tokenize_sequence(seq_to_translate)[::-1], input_vocab, lenght, with_pad=False)
    # prepare the 2D tensors to feed the model
    input_seq_len = torch.tensor([len(ids_seq_to_translate)], dtype=torch.int64)
    input_seq_len = input_seq_len.to(device)
    input_seq = torch.tensor(ids_seq_to_translate, dtype=torch.int64).unsqueeze(1)
    input_seq = input_seq.to(device)

    # set on evaluation mode
    model.eval()

    # run the forward pass of the encoder
    encoder_outputs, hidden, cell = model.encoder(input_seq, input_seq_len)

    # take the id value of <sos> from the target vocabulary
    sos_index = [target_vocab['<sos>'][0]]
    next_token = torch.LongTensor(sos_index).to(device)

    # initialize a tensor in the attention case to plot the attention weights later...
    if ATTENTION:
        attentions = torch.zeros(30, 1, len(ids_seq_to_translate)).to(device)

    # this list will contain the predicted words of the current sentence
    predicted_words = []

    # no gradient computations are needed
    with torch.no_grad():
        # In loop run the forward pass of the OneStepDecoder until some specified step (say 30)
        for i in range(30):
            output, hidden, cell, attention_weights = model.decoder.one_step_decoder(next_token, hidden, cell, encoder_outputs)

            # collecting the attention weights
            if ATTENTION:
                attentions[i] = attention_weights

            # take the most probable word
            next_token = output.argmax(1)

            # from id to word
            predicted_word = get_word(next_token.item(), target_vocab)

            # stop if the model reaches <eos>
            if predicted_word == '<eos>':
                break

            # appending the predicted word to the list
            predicted_words.append(predicted_word)

    # plot the attention weights
    if ATTENTION:
        display_attention(tokenize_sequence(seq_to_translate), predicted_words, attentions[:len(predicted_words)+1], att_plot_name)

    return ' '.join(predicted_words)


def display_attention(sentence, translation, attention, att_plot_name):
    '''
    Plot the weights of the attention mechanism.
    '''

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    attention = attention.squeeze(1).cpu().detach().numpy()[:-1, 1:-1]

    ax.matshow(attention, cmap='bone')

    ax.tick_params(labelsize=15)
    ax.set_xticklabels([''] + sentence + [''], rotation=45)
    ax.set_yticklabels([''] + translation + [''])

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.savefig(att_plot_name)


class Window(QDialog):
    '''
    Simple interface to run the `on_the_fly´ mode in local PC.
    '''

    def __init__(self, model, input_vocab, target_vocab):
        super().__init__()

        self.model = model
        self.input_vocab = input_vocab
        self.target_vocab = target_vocab

        # a figure instance to plot on
        self.figure = plt.figure()

        # this is the Canvas Widget that displays the `figure´
        self.canvas = FigureCanvas(self.figure)

        # this is the Navigation widget
        self.toolbar = NavigationToolbar(self.canvas, self)

        # some widgets
        self.button = QPushButton('Translate')
        self.button.clicked.connect(self.on_the_fly)
        self.input_seq = QLineEdit()
        self.output_seq = QLineEdit()

        # design the structure of the window
        hbox1 = QHBoxLayout()
        hbox1.addWidget(self.input_seq)
        hbox1.addWidget(self.button)
        hbox1.addWidget(self.output_seq)

        hbox2 = QHBoxLayout()
        hbox2.addWidget(self.toolbar)

        hbox3 = QHBoxLayout()
        hbox3.addWidget(self.canvas)

        layout = QVBoxLayout()
        layout.addLayout(hbox1)
        layout.addLayout(hbox2)
        layout.addLayout(hbox3)
        self.setLayout(layout)

        self.setGeometry(180,140,1600,1000)


    def on_the_fly(self):
        '''
        Translate a sentence on the fly.
        '''

        # we take the sequence to translate from the input QLineEdit
        seq_to_translate = self.input_seq.text().strip()
        if seq_to_translate == '':
            self.figure.clear()
            self.canvas.draw()
            self.output_seq.setText('')
            return

        lenght = len(tokenize_sequence(seq_to_translate))
        # we do not revert the input sentence in case of attention or bidirectional
        if ATTENTION or BIDIRECTIONAL:
            ids_seq_to_translate = ids_sequence(tokenize_sequence(seq_to_translate), self.input_vocab, lenght, with_pad=False)
        # we revert the input sentence in the basic case
        else:
            ids_seq_to_translate = ids_sequence(tokenize_sequence(seq_to_translate)[::-1], self.input_vocab, lenght, with_pad=False)
        # prepare the 2D tensors to feed the model
        input_seq_len = torch.tensor([len(ids_seq_to_translate)], dtype=torch.int64)
        input_seq_len = input_seq_len.to(device)
        input_seq = torch.tensor(ids_seq_to_translate, dtype=torch.int64).unsqueeze(1)
        input_seq = input_seq.to(device)

        # set on evaluation mode
        self.model.eval()

        # run the forward pass of the encoder
        encoder_outputs, hidden, cell = self.model.encoder(input_seq, input_seq_len)

        # take the id value of <sos> from the target vocabulary
        sos_index = [self.target_vocab['<sos>'][0]]
        next_token = torch.LongTensor(sos_index).to(device)

        # initialize a tensor in the attention case to plot the attention weights later...
        if ATTENTION:
            attentions = torch.zeros(30, 1, len(ids_seq_to_translate)).to(device)

        # this list will contain the predicted words of the current sentence
        predicted_words = []

        # no gradient computations are needed
        with torch.no_grad():
            # In loop run the forward pass of the OneStepDecoder until some specified step (say 30)
            for i in range(30):
                output, hidden, cell, attention_weights = self.model.decoder.one_step_decoder(next_token, hidden, cell, encoder_outputs)

                # collecting the attention weights
                if ATTENTION:
                    attentions[i] = attention_weights

                # take the most probable word
                next_token = output.argmax(1)

                # from id to word
                predicted_word = get_word(next_token.item(), self.target_vocab)

                # stop if the model reaches <eos>
                if predicted_word == '<eos>':
                    break

                # appending the predicted word to the list
                predicted_words.append(predicted_word)

        # plot the attention weights
        if ATTENTION:
            self.display_attention(tokenize_sequence(seq_to_translate), predicted_words, attentions[:len(predicted_words)+1])

        # we show the translated sequence in the output QLineEdit
        self.output_seq.setText(' '.join(predicted_words))


    def display_attention(self, sentence, translation, attention):
        '''
        Plot the weights of the attention mechanism.
        '''

        # clear the previous translation figure
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        attention = attention.squeeze(1).cpu().detach().numpy()[:-1, 1:-1]

        ax.matshow(attention, cmap='bone')

        ax.tick_params(labelsize=15)
        ax.set_xticklabels([''] + sentence + [''], rotation=45)
        ax.set_yticklabels([''] + translation + [''])

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        # draw on canvas instead of save the figure
        self.canvas.draw()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', help='specify the mode: train, evaluation or translation on the fly', choices = ['train', 'eval', 'on_the_fly'] )
    parser.add_argument('--run_on_colab', default='False', help='specify with a boolean if you are running the code on colab or not')
    parser.add_argument('--dataset_file', default='dataset/en_it.txt', help='specify the dataset file name, default: dataset/en_it.txt')
    parser.add_argument('--embed_file', help='specify the name of the input pretrained embedding file')
    parser.add_argument('--lr', default=0.001, help='specify the learning rate during train, default: 0.001')
    parser.add_argument('--embedding_dim', default=256, help='specify the embedding dimension of the model, default: 64')
    parser.add_argument('--hidden_dim', default=1024, help='specify the hidden dimension for both Encoder and OneStepDecoder (supposing same hidden dimension), default: 1024')
    parser.add_argument('--dropout', default=0.0, help='specify the probability of dropout during train, default: 0.1')
    parser.add_argument('--n_layers', default=1, help='specify the number of layers of the lstm for both Encoder and OneStepDecoder (supposing same hidden dimension), default: 2')
    parser.add_argument('--bidirectional', default='False', help='specify with a boolean if you want to use a bidirectional lstm for the Encoder, default=False')
    parser.add_argument('--attention', default='True', help='specify with a boolean if you want to use the attention mechanism, default=False')
    parser.add_argument('--teacher_forcing_ratio', default=0.5, help='specify the probability of teacher forcing ratio during train, default: 0.5')
    parser.add_argument('--epochs', default=10, help='specify the number of epochs during train, default: 10')
    parser.add_argument('--batch_size', default=64, help='specify the size of minibatches, default: 128')
    parser.add_argument('--frac', default=1, help='specify the integer by which divide the dataset if we want to use a subset of the entire dataset, default: 1 (the entire dataset)')
    parser.add_argument('--device', choices = ['cpu', 'gpu'], default='gpu', help='choose the device to use, default: gpu')
    parser.add_argument('--model', default='', help='specify the model to evaluate or translate on the fly')
    args = parser.parse_args()

    ########### ANALYZING THE COMMAND LINE INSTRUCTION ##############

    # the model name will be inferred during the training
    model_name = ''

    # check correctness for `run_on_colab´ argument
    if args.run_on_colab != 'True' and args.run_on_colab != 'False':
        raise Exception("run_on_colab must be `True´ or `False´")
    if args.run_on_colab == 'True':
        PREFIX_COLAB = '/content/drive/Shareddrives/project-Vitanza-Redi/'
        if args.mode == 'train':
            model_name += 'c_'

    # set right `PRETRAINED_EMBEDDINGS´ structure to use pretrained embeddings
    if args.embed_file is not None:
        embed_file = args.embed_file if args.embed_file.endswith(".txt") else args.embed_file + ".txt"
        PRETRAINED_EMBEDDINGS = [True, embed_file]
        model_name += 'pe_'

    # check correctness for `bidirectional´ argument
    if args.bidirectional != 'True' and args.bidirectional != 'False':
        raise Exception("bidirectional must be `True´ or `False´")
    if args.bidirectional == 'True':
        if float(args.dropout) != 0.0 or int(args.n_layers) != 1 or args.attention == 'True':
            raise Exception("You can use a bidirectional lstm with just one layer, dropout zero and no attention mechanism!")
        model_name += 'bi_'

    # check correctness for `attention´ argument
    if args.attention != 'True' and args.attention != 'False':
        raise Exception("attention must be `True´ or `False´")
    if args.attention == 'True':
        if float(args.dropout) != 0.0 or int(args.n_layers) != 1 or args.bidirectional == 'True':
            raise Exception("You can use attention mechanism with just one layer non-bidirectional lstm and dropout zero!")
        model_name += 'att_'

    # setting `bidirectional´ and `attention´ arguments
    BIDIRECTIONAL = True if args.bidirectional == 'True' else False
    ATTENTION = True if args.attention == 'True' else False

    # setting the hyperparameters
    LR = float(args.lr)
    EMBEDDING_DIM = int(args.embedding_dim) if not PRETRAINED_EMBEDDINGS[0] else guess_embed_size(PREFIX_COLAB + "wembedds/" + PRETRAINED_EMBEDDINGS[1])
    HIDDEN_DIM = int(args.hidden_dim)
    N_LAYERS = int(args.n_layers)
    EPOCHS = int(args.epochs)
    BATCH_SIZE = int(args.batch_size)
    FRAC = int(args.frac)

    # check correctness for `dropout´ argument
    if float(args.dropout) < 0 or float(args.dropout) > 1:
        raise Exception('dropout is a probability between 0 and 1!')
    DROPOUT = float(args.dropout)

    # check correctness for `teacher_forcing_ratio´ argument
    if float(args.teacher_forcing_ratio) < 0 or float(args.teacher_forcing_ratio) > 1:
        raise Exception('teacher forcing ratio is a probability between 0 and 1!')
    TEACHER_FORCING_RATIO = float(args.teacher_forcing_ratio)

    # setting the right path for the dataset file
    dataset_file  = PREFIX_COLAB + args.dataset_file

    # finilize the model name in case of train
    if args.mode == 'train':
        model_name += f'lr{LR}_ed{EMBEDDING_DIM}_hd{HIDDEN_DIM}_dr{DROPOUT}_nl{N_LAYERS}_tfr{TEACHER_FORCING_RATIO}_e{EPOCHS}_bs{BATCH_SIZE}_frac{args.frac}.pth'
    # if mode is either `eval´ or `on_the_fly´ the --model argument must be specified
    elif args.model is None:
        raise Exception('Please, specify the model you want to use')
    else:
        model_name += args.model if args.model.endswith('.pth') else args.model+'.pth'

    # setting the chosen device
    if args.device == 'gpu':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'

    print('Running on Colab...') if PREFIX_COLAB != "" else print('Running on a local PC...')
    print("GPU available!") if device == 'cuda:0' else print('No GPU available, sorry :(')

    # training case...
    if args.mode == 'train':

        # Summarize the settings...
        print('Pretrained embedding: YES') if PRETRAINED_EMBEDDINGS[0] else print('Pretrained embedding: NO')
        print('Encoder\'s LSTM bidirectional: YES') if BIDIRECTIONAL else print('Encoder\'s LSTM bidirectional: NO')
        print('Attention mechanism: YES') if ATTENTION else print('Attention mechanism: NO')
        print(f'Embedding dimension: {EMBEDDING_DIM}')
        print(f'Encoder\'s hidden dimension:: {HIDDEN_DIM}')
        print(f'Decoder\'s hidden dimension:: {HIDDEN_DIM}') if not BIDIRECTIONAL else print(f'Decoder\'s hidden dimension:: {2 * HIDDEN_DIM}')
        print(f'LSTM\'s layers: {N_LAYERS}')
        print(f'Dropout probability: {DROPOUT}')
        print(f'Teacher-forcing-ratio probability: {TEACHER_FORCING_RATIO}')
        print(f'Learning rate: {LR}')
        print(f'Epochs: {EPOCHS}')
        print(f'Batch size: {BATCH_SIZE}')
        print(f'Percentage of dataset\'s usage: {round(100/FRAC, 2)}%')

        # reading the dataset file, eventually reduced using frac
        dataset_list = build_dataset_list(dataset_file)
        if FRAC != 1:
            dataset_list = [dataset_list[i] for i in range(0,len(dataset_list)) if i % FRAC == 0]

        # separating data by language
        sequences_eng = [row[0] for row in dataset_list]
        sequences_ita = [row[1] for row in dataset_list]

        # check on the dataset folder if the splits already exist
        if not os.path.isdir(PREFIX_COLAB + f"dataset/splits_frac_{FRAC}/"):
            os.makedirs(PREFIX_COLAB + f'dataset/splits_frac_{FRAC}/')
            print("Creating splits...")
            train_set, val_set = create_splits(dataset_list, 0.7, 0.15)
        else:
            print("Splits already exist!")
            train_set, val_set, _ = load_dataset_splits()

        # creating the vocabulary files
        print("Building eng vocabulary...")
        vocab_eng = create_vocabulary(sequences_eng)
        with open(PREFIX_COLAB + "vocabularies/vocab_eng.txt", "w") as f:
            for k, v in vocab_eng.items():
                f.write(f"{v[0]}  {k}  {v[1]}\n")

        print("Building ita vocabulary...")
        vocab_ita = create_vocabulary(sequences_ita)
        with open(PREFIX_COLAB + "vocabularies/vocab_ita.txt", "w") as f:
            for k, v in vocab_ita.items():
                f.write(f"{v[0]}  {k}  {v[1]}\n")

        print("Starting to train the model...")
        # train the model and return the best model, the best epoch and the losses for each epoch (to plot)
        model, best_epoch, loss_train, loss_val = train(train_set, val_set, vocab_eng, vocab_ita, EPOCHS, BATCH_SIZE)

        # saving the checkpoint
        checkpoint = {
                'model_state_dict': model.state_dict(),
                'input_vocab': vocab_eng,
                'target_vocab': vocab_ita,
                'ed': EMBEDDING_DIM,
                'hd': HIDDEN_DIM,
                'nl': N_LAYERS,
                'dr': DROPOUT,
                'bi': BIDIRECTIONAL,
                'att': ATTENTION,
                'frac': FRAC}

        torch.save(checkpoint, PREFIX_COLAB + "models/" + model_name)

        print(f'Best model found at epoch {best_epoch} --- train loss {loss_train[best_epoch-1]} --- val loss {loss_val[best_epoch-1]}')
        print(f'Model saved as {PREFIX_COLAB}models/{model_name}')

        # saving the plot
        plot(loss_train, loss_val, PREFIX_COLAB + "plots/" + model_name[:-4] + "_plot.pdf")

    # evaluation case...
    elif args.mode == 'eval':

        # setting the best model if no specified
        if args.model == '':
            model_name = 'c_att_lr0.001_ed256_hd1024_dr0.0_nl1_tfr0.5_e10_bs64_frac1.pth'
        # load the checkpoint to rebuild the trained model
        # avoid error: if you are running on a CPU-only machine, please use torch.load 
        # with map_location=torch.device('cpu') to map your storages to the CPU.
        if device=='cpu':
        	checkpoint = torch.load(PREFIX_COLAB + "models/" + model_name, map_location=torch.device('cpu'))
        else:
        	checkpoint = torch.load(PREFIX_COLAB + "models/" + model_name)
        input_vocab = checkpoint['input_vocab']
        target_vocab = checkpoint['target_vocab']
        EMBEDDING_DIM = checkpoint['ed']
        HIDDEN_DIM = checkpoint['hd']
        N_LAYERS = checkpoint['nl']
        DROPOUT = checkpoint['dr']
        BIDIRECTIONAL = checkpoint['bi']
        ATTENTION = checkpoint['att']
        FRAC = checkpoint['frac']
        model = create_model(input_vocab, target_vocab, 'eval')
        model.load_state_dict(checkpoint['model_state_dict'])

        # load the test set
        _, _, test_set = load_dataset_splits()

        # test the translation
        test(test_set, model, input_vocab, target_vocab, PREFIX_COLAB + "tests/" + model_name[:-4] + "_test.txt")

    # on the fly case...
    elif args.mode == 'on_the_fly':

        # setting the best model if no specified
        if args.model == '':
            model_name = 'c_att_lr0.001_ed256_hd1024_dr0.0_nl1_tfr0.5_e10_bs64_frac1.pth'
        # load the checkpoint to rebuild the trained model
        # avoid error: if you are running on a CPU-only machine, please use torch.load 
        # with map_location=torch.device('cpu') to map your storages to the CPU.
        if device=='cpu':
        	checkpoint = torch.load(PREFIX_COLAB + "models/" + model_name, map_location=torch.device('cpu'))
        else:
        	checkpoint = torch.load(PREFIX_COLAB + "models/" + model_name)
        input_vocab = checkpoint['input_vocab']
        target_vocab = checkpoint['target_vocab']
        EMBEDDING_DIM = checkpoint['ed']
        HIDDEN_DIM = checkpoint['hd']
        N_LAYERS = checkpoint['nl']
        DROPOUT = checkpoint['dr']
        BIDIRECTIONAL = checkpoint['bi']
        ATTENTION = checkpoint['att']
        FRAC = checkpoint['frac']
        model = create_model(input_vocab, target_vocab, 'on_the_fly')
        model.load_state_dict(checkpoint['model_state_dict'])

        # case of colab...
        if PREFIX_COLAB != "":
            # on the fly translation
            seq_to_translate = input('Please, enter the sentence you want to translate:\n')
            seq_translated = on_the_fly(seq_to_translate, model, input_vocab, target_vocab, PREFIX_COLAB + "att_plots/" + model_name[:-4] + "_att_plot.pdf")
            print(f"Translated sentence: {seq_translated}")

        # case of local PC using PyQt interface...
        else:
            app = QApplication(["Translator"])
            win = Window(model, input_vocab, target_vocab)
            win.show()
            sys.exit(app.exec_())
