import pandas as pd
import numpy as np

class Set:
    def set_training(self, filename):
        self.training = pd.read_table(filename, sep=' ', names=['words', 'tags'])


class HMM:
    def __init__(self, training_dataset=None):
        self.training_set = training_dataset.training
        self.tags = self.training_set.tags.unique()
        self.words = self.training_set.words.unique()

    def set_training_set(self, training_dataset):
        self.training_set = training_dataset.data

    def count_y_to_x(self, x, y):
        df = self.training_set
        df = df[df['words']==x]
        df = df[df['tags']==y]
        count = df.shape[0]
        return count

    def count_y(self, y):
        df = self.training_set
        df = df[df['tags'] == y]
        count = df.shape[0]
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

    def set_params(self, df):
        self.params = df

    def generate_tag(self, input_filename, output_filename=None):
        f = open(output_filename, 'w')
        input_file = open(input_filename, 'r')

        words = [x.strip() for x in input_file.readlines()]

        for i in range(len(words)):
            x = words[i]
            if x == '':
                f.write("\n")
            else:
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
hmm = HMM(d)

## Uncomment to train model and save parameters
# hmm.train()
# hmm.params.to_pickle("./EN/params.pkl")

## Uncomment to load trained parameters
df = pd.read_pickle("./EN/params.pkl")
hmm.set_params(df)

hmm.generate_tag(input_filename="./EN/dev.in", output_filename='./EN/dev.p2.out')