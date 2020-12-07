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

        f.close()


class HMM:
    def __init__(self, training_dataset=None):
        self.training_set = training_dataset.training
        self.tags = self.training_set.tags.unique()
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
        print('training...')
        for tag in self.tags:

            prob = []
            for next_tag in self.tags:
                q = self.trans_params(tag, next_tag)
                prob.append(q)

            yparams.append(prob)
            y.append(str(tag))

        df = pd.DataFrame({'tags': y, 'y_params': yparams}, columns={'tags', 'y_params'})
        self.transi_params = df

    def set_params(self, dfx, dfy):
        self.emission_params = dfx
        self.transistion_params = dfy

    def max_b(self, word):
        x = word

        if x in self.words:
            x1 = x
        else:
            x1 = '#UNK#'

        row = self.emission_params.loc[self.emission_params['words'] == x1]
        probs = row['params'].values[0]

        return probs

    def max_a(self, prev_y):
        row = self.transistion_params.loc[self.transistion_params['tags'] == prev_y]
        probs = row['y_params'].values[0]

        return probs

    def viterbi(self, in_file, out_file):
        f_in = open(in_file, 'r', encoding="utf-8")
        f_out = open(out_file, 'w', encoding="utf-8")
        print('Viterbi running...')

        words = ['\n']
        lines = f_in.readlines()
        for i in lines:
            if i == '\n':
                words.append(i)
            else:
                words.append(i.strip())

        newtags_list = []

        for i in range(len(words)):
            if words[i] == '\n':
                scores = []
                pi_j = 1
                some_tag = 'S'
                for j in range(i, len(words) - 1):
                    a_probs = self.max_a(some_tag)
                    b_probs = self.max_b(words[j+1])
                    ab = []
                    for k, l in zip(a_probs, b_probs):
                        ab.append(k*l)
                    pi_j = pi_j * max(ab)
                    pos = np.argmax(ab)
                    new_tag = self.tags[pos]
                    newtags_list.append(new_tag)

                    some_tag = new_tag
                    scores.append(pi_j)

                    if words[j+2] == '\n':
                        last_score = max(self.max_a(some_tag))
                        pi_n1 = pi_j * last_score
                        scores.append(pi_n1)
                        pos = np.argmax(self.max_a(some_tag))
                        newtags_list.append(self.tags[pos])
                        break

        generated_tags = ['' if k == 'S' else k for k in newtags_list]
        # new_list = []
        # to_write = []
        # for i, j in zip(words[1:-1], generated_tags):
        #     new_list.append('{} {}'.format(i, j))
        # for k in new_list:
        #     if k != ' ':
        #         to_write.append(k)
        #
        # for i in to_write:
        #     f_out.write('{}\n'.format(i))

        for i, j in zip(words[1:], generated_tags):
            # f_out.write('{} {}\n'.format(i, j))
            if i == '\n':
                f_out.write('{}'.format(i))
            else:
                f_out.write('{} {}\n'.format(i, j))

        f_out.close()
        f_in.close()


d = Set()
d.set_training('./EN/train')
hmm = HMM(d)

# To train model and save parameters
hmm.train_trans_params()
hmm.transi_params.to_pickle("./EN/y_params.pkl")

# Load trained parameters
df_x = pd.read_pickle("./EN/params.pkl")
df_y = pd.read_pickle("./EN/y_params.pkl")
hmm.set_params(df_x, df_y)

hmm.viterbi("./EN/dev.in", './EN/dev.p3.out')
