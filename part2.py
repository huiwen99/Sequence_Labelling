import pandas as pd
import numpy as np

class Set:
    def set_training(self, filename):
        self.training = pd.read_table(filename, sep=' ', names=['words', 'tags'])

    def set_dev(self, filename):
        self.dev = pd.read_table(filename, names=['words'])


class HMM:
    def __init__(self, training_dataset=None):
        self.training_set = training_dataset.training
        self.tags = self.training_set.tags.unique()
        self.words = self.training_set.words.unique()

    def set_training_set(self, training_dataset):
        self.training_set = training_dataset.data

    def count_y_to_x(self, x, y):
        count = 0
        for i in range(self.training_set.shape[0]):
            word = self.training_set['words'][i]
            tag = self.training_set['tags'][i]
            if word == x and y == tag:
                count += 1
        return count

    def count_y(self, y):
        count = 0
        for i in range(self.training_set.shape[0]):
            tag = self.training_set['tags'][i]
            if y == tag:
                count += 1
        return count

    def emission_params(self, x, y, k=0.5):
        if x=='#UNK#':
            e = k / (self.count_y(y) + k)
        else:
            num = self.count_y_to_x(x, y)
            den = self.count_y(y) + k
            e = num / den
        return e

    def train(self):
        x = []
        params = []
        for word in self.words:
            probs = []
            for tag in self.tags:
                e = self.emission_params(word, tag)
                probs.append(e)
            x.append(word)
            params.append(probs)


        probs = []
        for tag in self.tags:
            e = self.emission_params('#UNK#',tag)
            probs.append(e)
        x.append('#UNK#')
        params.append(probs)

        df = pd.DataFrame({'words':x, 'params':params}, columns = ['words','params'])
        self.params = df

    def generate_tag(self, words_df, output_filename=None):
        f = open(output_filename, 'w')

        for i in range(words_df.shape[0]):
            x = words_df['words'][i]
            if x in self.words:
                x1 = x
            else:
                x1 = '#UNK#'
            row = self.params.loc[self.params['words']==x1]
            probs = row['params'].values[0]
            pos = np.argmax(probs)
            y = self.tags[pos]

            f.write("{} {}\n".format(x, y))

        f.close()




d = Set()
d.set_training('./EN/train')
d.set_dev('./EN/dev.in')
hmm = HMM(d)
print(hmm.tags)
hmm.train()
hmm.generate_tag(d.dev, output_filename='./EN/dev.p2.out')