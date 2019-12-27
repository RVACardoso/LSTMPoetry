import numpy as np
np.set_printoptions(threshold=np.nan)
import pandas as pd
from keras.utils import np_utils
import unidecode
from sklearn import preprocessing

punct_dict = {
        '.': '||period||',
        ',': '||comma||',
        '"': '||quotes||',
        ';': '||semicolon||',
        '!': '||exclamation-mark||',
        '?': '||question-mark||',
        '(': '||left-parentheses||',
        ')': '||right-parentheses||',
        '--': '||emm-dash||',
        '\n': '||return||'}


class TextMapping:
    def __init__(self, file="file.txt"):
        raw_text = open(file).read().lower()
        self.ascii_text = unidecode.unidecode(raw_text)


    def character_map(self, seq_size=50):  #FOR CAMOES
        self.origin = 'character'

        self.charac = sorted(list(set(self.ascii_text)))
        self.nr_to_letters = {nr: char for nr, char in enumerate(self.charac)}
        self.letters_to_nr = {char: nr for nr, char in enumerate(self.charac)}
        X, Y = [], []
        for i in range(0, len(self.ascii_text) - seq_size, 1):
            sequence = self.ascii_text[i:i + seq_size]
            label = self.ascii_text[i + seq_size]
            X.append([self.letters_to_nr[char] for char in sequence])
            Y.append(self.letters_to_nr[label])

        X_modified = np.reshape(X, (len(X), seq_size, 1))
        #X_modified = X_modified / float(len(self.charac))

        self.le_charac = preprocessing.LabelEncoder()
        self.le_charac.fit(np.array(X).flatten())
        Y_modified = self.le_charac.transform(Y)

        return X_modified, Y_modified


    def word_map(self, seq_size=10): # FOR PESSOA AND CESARIO VERDE
        self.origin = 'word'

        token_text = self.ascii_text
        for punc, replacement in punct_dict.items():
            token_text = token_text.replace(punc, ' {} '.format(replacement))
        token_text = token_text.split()

        self.vocab = sorted(set(token_text))
        self.nr_to_letters = {nr: word for nr, word in enumerate(self.vocab)}
        self.letters_to_nr = {word: nr for nr, word in enumerate(self.vocab)}

        X, Y = [], []
        for i in range(0, len(token_text) - seq_size, 1):
            sequence = token_text[i:i + seq_size]
            label = token_text[i + seq_size]
            X.append([self.letters_to_nr[word] for word in sequence])
            Y.append(self.letters_to_nr[label])

        X_modified = np.reshape(X, (len(X), seq_size, 1))
        X_modified = (2.0*X_modified / float(len(self.vocab))) - 1.0

        # Y_modified = np_utils.to_categorical(Y)
        self.le_word = preprocessing.LabelEncoder()
        self.le_word.fit(np.array(X).flatten())
        Y_modified = self.le_word.transform(Y)

        return X_modified, Y_modified


    # def one_hot_feat(self, X):
    #     if self.origin == 'word':
    #         length = len(self.vocab)
    #     elif self.origin == 'character':
    #         length = len(self.charac)
    #
    #     X_one_hot = np.array([])
    #     for j in range(X.shape[0]):
    #         print("seq_number: {}".format(j))
    #         for i in range(X.shape[1]):
    #             line = np.zeros((length))
    #             line[X[j][i]] = 1.0
    #             X_one_hot = np.append(X_one_hot, [line])
    #
    #     X_one_hot = np.reshape(X_one_hot, (X.shape[0], X.shape[1], length))
    #
    #     return X_one_hot
    #
    #
    # def fast_one_hot_feat(self, X):
    #     if self.origin == 'word':
    #         length = len(self.vocab)
    #     elif self.origin == 'character':
    #         length = len(self.charac)
    #
    #     sequence_size = X.shape[1]
    #
    #     X_one_hot = []
    #     # print(X.shape)
    #     for i in range(sequence_size):
    #         line = np.zeros((length))
    #         line[X[0][i]] = 1.0
    #         X_one_hot = np.append(X_one_hot, [line])
    #
    #
    #     X_one_hot = np.array([X_one_hot]).flatten()
    #     # print(X_one_hot.shape)
    #     for j in range(1, X.shape[0]):
    #         if j%10 == 0:
    #             print(j)
    #         X_one_hot = np.append(X_one_hot, X_one_hot[-(sequence_size-1)*length:])
    #         line = np.zeros((length))
    #         line[X[j][sequence_size-1]] = 1.0
    #         X_one_hot = np.append(X_one_hot, [line])
    #         #print(X_one_hot.shape)
    #
    #     X_one_hot = np.reshape(X_one_hot, (X.shape[0], X.shape[1], length))
    #     # print(X_one_hot.shape)
    #
    #     return X_one_hot

    def faster_one_hot_feat(self, X):
        if self.origin == 'word':
            length = len(self.vocab)
        elif self.origin == 'character':
            length = len(self.charac)

        sequence_size = X.shape[1]

        seq1_one_hot = []
        # print(X.shape)
        for i in range(sequence_size):
            line = np.zeros((length))
            line[X[0][i]] = 1.0
            seq1_one_hot = np.append(seq1_one_hot, [line])
        seq_one_hot = np.reshape(seq1_one_hot, (sequence_size, length))

        #print(seq_one_hot.shape)
        line_set = []
        for j in range(1, X.shape[0]):
            # if j % 10 == 0:
            #     print('loop: {}/{}'.format(j, X.shape[0]))
            line = np.zeros((length))
            line[X[j][sequence_size - 1]] = 1.0
            line_set.append(line)

        line_set = np.array(line_set)
        seq_one_hot = np.concatenate((seq_one_hot, line_set))

        # print("getting sequences...")
        X_one_hot = []
        for j in range(X.shape[0]):
            # if j % 10 == 0:
            #     print('seq: {}/{}'.format(j, X.shape[0]))
            X_one_hot.append(seq_one_hot[j:j+sequence_size])

        # print("final reshape...")
        X_one_hot = np.array(X_one_hot, dtype=np.byte)
        X_one_hot = np.reshape(X_one_hot, (X.shape[0], X.shape[1], length))

        return X_one_hot

    def rev_one_hot_feat(self, X_one_hot):
        X_rev_one_hot = []
        if len(X_one_hot.shape) == 2:
            X_one_hot = np.reshape(X_one_hot, (1, X_one_hot.shape[0], X_one_hot.shape[1]))

        for l in range(X_one_hot.shape[0]):
            for k in range(X_one_hot.shape[1]):
                X_rev_one_hot.append(np.argmax(X_one_hot[l][k]))

        X_rev_one_hot = np.array(X_rev_one_hot)
        X_rev_one_hot = np.reshape(X_rev_one_hot, (X_one_hot.shape[0], X_one_hot.shape[1], 1))

        return X_rev_one_hot

    # backup
    def one_hot_label(self, X):
        if self.origin == 'word':
            length = len(self.vocab)
        elif self.origin == 'character':
            length = len(self.charac)

        line_set = []
        for j in range(X.shape[0]):
            line = np.zeros((length))
            line[X[j]] = 1.0
            line_set.append(line)

        X_one_hot = np.concatenate(line_set)

        X_one_hot = np.reshape(X_one_hot, (X.shape[0], length))
        return X_one_hot


    # def one_hot_label(self, X):
    #     if self.origin == 'word':
    #         length = len(self.vocab)
    #     elif self.origin == 'character':
    #         length = len(self.charac)
    #
    #     line_set = []
    #     for j in range(X.shape[0]):
    #         line = np.zeros((length))
    #         line[X[j]] = 1.0
    #         line_set.append(np.array(line, dtype=np.byte))
    #
    #     X_one_hot = np.concatenate(np.array(line_set, dtype=np.byte))
    #
    #     X_one_hot = np.reshape(X_one_hot, (X.shape[0], length))
    #     return X_one_hot


    def rev_one_hot_single_label(self, X_one_hot):
        return np.argmax(X_one_hot)


    def seed_char_to_text(self, X_one_hot):
        chars = []
        for nr in self.rev_one_hot_feat(X_one_hot)[0].flatten():
            chars.append(self.nr_to_letters[nr])

        return ''.join(chars)


    def seed_word_to_text(self, X):
        chars = []
        for nr in X.flatten():
            chars.append(self.nr_to_letters[int(round((nr+1.0)*0.5*float(len(self.vocab))))])

        return self.get_word_punctuation(chars)


    def get_word_punctuation(self, chars):
        chars = ' '.join(chars)
        for punc, replacement in punct_dict.items():
            chars = chars.replace(' ' + replacement, '{}'.format(punc))
        return chars


    def generate_text_character(self, model, seed, out_size):
        # generating characters

        out = self.rev_one_hot_feat(seed)[0]
        out = [number[0] for number in out]
        out = [self.nr_to_letters[value] for value in out]

        seed = seed[0]
        for i in range(out_size):
            seed = np.array([seed])
            #print(seed.shape)
            pred_class = model.predict_classes(seed)
            #print(pred_class)
            pred_one_hot = np.zeros(seed.shape[2])
            pred_one_hot[pred_class[0]] = 1.0
            old_seed = seed[0]
            #print(old_seed)
            seed = np.vstack((old_seed[1:], pred_one_hot))

            new_nr = self.rev_one_hot_single_label(pred_one_hot)
            new_char = self.nr_to_letters[new_nr]
            out.append(new_char)

        return ''.join(out)


    def generate_text_word(self, model, seed, out_size):
        # generating characters
        print(self.nr_to_letters)

        out = [number[0] for number in seed[0]]
        print(out)

        out = [self.nr_to_letters[int(round((value+1.0)*0.5*float(len(self.vocab))))] for value in out]
        print(out)


        seed = seed[0]
        for i in range(out_size):
            seed = np.array([seed])
            #print(seed)
            pred_nr = model.predict_classes(seed)
            pred_small = (2.0*pred_nr[0] / float(len(self.vocab))) - 1.0
            # (2.0*X_modified / float(len(self.vocab))) - 1.0
            old_seed = seed[0]

            seed = np.vstack((old_seed[1:], [pred_small]))
            print(seed)
            print(pred_small)

            new_char = self.nr_to_letters[pred_nr[0]]
            out.append(new_char)

        return self.get_word_punctuation(out)


#TESTS

# map1 = TextMapping(file="file.txt")
# # X, Y = map1.character_map()
# X, Y = map1.word_map()
# # idx=51
# # print(Y[idx])
#
#
# X_one_hot = map1.faster_one_hot_feat(X)
# Y_one_hot = map1.one_hot_label(Y)
#
# ## print(Y_one_hot[idx])
#
# # X_rev_one_hot = map1.rev_one_hot_feat(X_one_hot)
# # Y_rev_one_hot = map1.rev_one_hot_single_label(Y_one_hot[idx])
# # print(X_rev_one_hot[32])
# # print(X_rev_one_hot.shape)
#
# # print("{}   {}".format(Y[idx], Y_rev_one_hot))
#
# print('seed: {}'.format(map1.seed_to_text(X_one_hot[0])))
# out = map1.generate_text(model=Y_one_hot, seed=X_one_hot[0], out_size=5)
# print('outp: {}'.format(out))