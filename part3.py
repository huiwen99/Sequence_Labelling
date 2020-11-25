import pandas as pd
import numpy as np


class Set:
    def set_training(self, filename):
        f = open(filename, 'r', encoding='utf8')
        lines = f.readlines()

        words = []
        tags = []

        for line in lines:
            if len(line) != 1:
                word, tag = line.split()
                words.append(word)
                tags.append(tag)
            else:
                tags.append('S')
                words.append('\n')

        words = np.array(words)
        tags = np.array(tags)
        df = pd.DataFrame({'words': words, 'tags': tags}, columns=['words', 'tags'])
        self.training = df


class HMM:
    def __init__(self, training_dataset=None):
        self.training_set = training_dataset.training
        tg = self.training_set.tags.unique()
        self.tags = np.append(tg, 'S')
        self.words = self.training_set.words.unique()

    def set_training_set(self, training_dataset):
        self.training_set = training_dataset.data

    # To estimate the transition parameters
    def count_y_to_y(self, prev_y, y):
        df = self.training_set
        count = 0
        for i in range(len(df['tags']) - 1):
            if df['tags'][i] == prev_y and df['tags'][i + 1] == y:
                count += 1
        return count

    def count_prev_y(self, prev_y):
        df = self.training_set
        count = 0
        for i in range(len(df['tags'])):
            if df['tags'][i] == prev_y:
                count += 1
        return count

    def trans_params(self, prev_y, y):
        num = self.count_y_to_y(prev_y, y)
        den = self.count_prev_y(prev_y)
        q = num / den
        return q

    def train_trans_params(self):
        yparams = []
        y = []
        for tag in self.tags:
            prob = []
            given_prev_y = []
            for next_tag in self.tags:
                q = self.trans_params(tag, next_tag)
                prob.append(q)
                given_prev_y.append(next_tag)
            yparams.append(prob)
            y.append(given_prev_y)

        # x = []
        # for i in range(len(self.tags)-1):
        #     pair = [self.words[i], self.words[i+1]]
        #     x.append(pair)
        # y = []
        # for i in self.tags:
        #     y.append(i)

        df = pd.DataFrame({'tags:': y, 'y_params': yparams}, columns={'tags', 'y_params'})
        # df = pd.DataFrame({'words': x, 'tags:': y, 'y_params': yparams}, columns={'words', 'tags', 'y_params'})
        self.transistion_params = df

    def set_params(self, dfx, dfy):
        self.emission_params = dfx
        self.transistion_params = dfy

    def max_b(self, word):
        x = word
        if x == '':
            max_b_val = 1
        else:
            if x in self.words:
                x1 = x
            else:
                x1 = '#UNK#'
            row = self.emission_params.loc[self.emission_params['words'] == x1]
            probs = row['params'].values[0]
            max_b_val = max(probs)
            # pos = np.argmax(probs)
            # y = self.tags[pos]
        return max_b_val

    def max_a(self, prev_y):
        # if x in self.transistion_params['words']:
        row = self.transistion_params.loc[self.transistion_params['tags'] == prev_y]
        probs = row['y_params'].values[0]
        pos = np.argmax(probs)
        y = self.tags[pos]
        return max(probs), y

    def viterbi(self, in_file, out_file):
        f_in = open(in_file, 'r')
        f_out = open(out_file, 'w')
        # words = [x.strip() for x in f_in.readlines()]
        words = ['\n']
        lines = f_in.readlines()
        for i in lines:
            if i == '\n':
                words.append(i)
            else:
                words.append(i.strip())

        gen_tag = list(words)

        for i in range(len(words)):
            if words[i] == '\n':

                pi_j = 1
                for j in range(i, len(words) - 1):
                    b_score = self.max_b(words[j + 1])
                    a_score, new_tag = self.max_a(words[j])
                    pi_j = pi_j * b_score * a_score
                    gen_tag[j] = new_tag

                last_score, stop_tag = self.max_a(words[len(words)])
                pi_n1 = pi_j * last_score

        new_tags = ['' if k == 'S' else k for k in gen_tag]
        for i, j in zip(f_in.readlines(), new_tags):
            f_out.write('{} {}\n'.format(i, j))


d = Set()
d.set_training('./CN/train')
hmm = HMM(d)

# To train model and save parameters
hmm.train_trans_params()
hmm.transistion_params.to_pickle("./CN/y_params.pkl")

# Load trained parameters
df_x = pd.read_pickle("./CN/params.pkl")
df_y = pd.read_pickle("./CN/y_params.pkl")
hmm.set_params(df_x, df_y)

hmm.viterbi("./CN/dev.in", './CN/dev.p3.out')
