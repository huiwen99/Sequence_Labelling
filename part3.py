import pandas as pd
import numpy as np
import sys

class Set:
    def set_training(self, filename):
        f = open(filename, 'r', encoding='utf8')
        lines = f.readlines()

        words = []
        tags = []

        sword = []
        stag = []
        for line in lines:
            if len(line) != 1:
                word, tag = line.split()
                sword.append(word)
                stag.append(tag)
            else:
                stag.append('S')
                sword.append('\n')
        sword = np.array(sword)
        stag = np.array(stag)
        df_with_s = pd.DataFrame({'s_words': sword, 's_tags': stag}, columns={'s_words', 's_tags'})
        self.training_with_s = df_with_s

        for line in lines:
            if len(line) != 1:
                word, tag = line.split()
                words.append(word)
                tags.append(tag)

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

        self.s_training_set = training_dataset.training_with_s
        self.tags_with_s = self.s_training_set.s_tags.unique()

    def set_training_set(self, training_dataset):
        self.training_set = training_dataset.training
        self.s_training_set = training_dataset.training_with_s

    # To estimate the transition parameters
    def count_y_to_y(self, prev_y, y):
        df = self.s_training_set
        count = 0
        for i in range(len(df['s_tags']) - 1):
            if df['s_tags'][i] == prev_y and df['s_tags'][i + 1] == y:
                count += 1
        return count

    def count_prev_y(self, prev_y):
        df = self.s_training_set
        count = 0
        for i in range(len(df['s_tags'])):
            if df['s_tags'][i] == prev_y:
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
        for tag in self.tags_with_s:

            prob = []
            for next_tag in self.tags_with_s:
                q = self.trans_params(tag, next_tag)
                prob.append(q)

            yparams.append(prob)
            y.append(str(tag))

        df = pd.DataFrame({'tags': y, 'y_params': yparams}, columns={'tags', 'y_params'})
        self.transi_params = df

    def set_params(self, dfx, dfy):
        self.emission_params = dfx
        self.transistion_params = dfy

    def get_sentence(self, in_file):
        f_in = open(in_file, 'r', encoding="utf-8")
        sentence = []
        lines = f_in.readlines()

        words = ['\n']
        for line in lines:
            if line == '\n':
                words.append('\n')
                sentence.append(words)
                words = ['\n']
            else:
                words.append(line.strip())
        f_in.close()
        return sentence

    def pi(self,k,v,x,stop=False):
        if k==0:
            if v=='S':
                return 1
            else:
                return 0
        else:
            values = []
            for i in range(self.transistion_params.tags.shape[0]):
                u = self.transistion_params.tags[i]
                value = self.pi_values_all_steps[k-1][i]

                if stop==False:
                    if v=='S':
                        b=0
                    else:

                        df = self.emission_params
                        emi = df.loc[df['words']==x]['params'].values[0]
                        pos = list(self.tags).index(v)
                        b = emi[pos]
                else:
                    b = 1

                df = self.transistion_params
                trans = df.loc[df['tags']==u]['y_params'].values[0]
                pos = list(self.transistion_params.tags).index(v)
                a = trans[pos]

                value = value * a * b
                values.append(value)
            pi_value = max(values)
            return pi_value

    def find_tag(self,k,next_tag):
        values = []
        for v in self.transistion_params.tags:
            pos = list(self.transistion_params.tags).index(v)
            pi = self.pi_values_all_steps[k][pos]

            df = self.transistion_params
            trans = df.loc[df['tags'] == v]['y_params'].values[0]
            pos = list(self.transistion_params.tags).index(next_tag)
            a = trans[pos]

            value = pi*a
            values.append(value)
        pos = np.argmax(values)
        tag = self.transistion_params.iloc[pos].tags
        return tag

    def viterbi2(self, in_file, out_file):
        sentences = self.get_sentence(in_file)
        f_out = open(out_file, 'w', encoding="utf-8")

        for sentence in sentences:
            n = len(sentence)

            ## forward pass
            self.pi_values_all_steps = []

            for k in range(n-1):
                pi_values = []
                x = sentence[k]
                if x not in self.words:
                    x = '#UNK#'
                for v in self.transistion_params.tags:
                    pi = self.pi(k,v,x)
                    pi_values.append(pi)
                self.pi_values_all_steps.append(pi_values)


            # stop case
            pi_values = []
            for v in self.transistion_params.tags:
                pi = self.pi(n-1, v, sentence[n-1], stop=True)
                pi_values.append(pi)
            self.pi_values_all_steps.append(pi_values)

            ## backward pass

            tags = ['S']
            for k in range(n-2,-1,-1):
                next_tag = tags[0]
                tag = self.find_tag(k,next_tag)
                tags.insert(0,tag)

            sentence_to_write = sentence[1:-1]
            tags_to_write = tags[1:-1]

            for i in range(len(sentence_to_write)):
                x = sentence_to_write[i]
                y = tags_to_write[i]
                f_out.write("{} {}\n".format(x, y))

            f_out.write("\n")


if __name__=="__main__":
    if len(sys.argv < 3):
        print("Make sure at least python 3.8 is installed")
        print("Run the file in this format")
        print("python part2.py [dataset] [mode]")
        print("dataset can be EN,SG,CN") # sys.argv[1]
        print("mode can be train or predict") # sys.argv[2]

    else:
        dataset = sys.argv[1]
        mode = sys.argv[2]

        hmm.dataset = dataset

        d = Set()
        d.set_training('./{}/train'.format(dataset))
        hmm = HMM(d)

        if mode == "train":
            print("Training parameters")
            hmm.train_trans_params()
            hmm.transi_params.to_pickle("./{}/y_params.pkl".format(dataset))
            print("Parameters is saved to ./{}/y_params.pkl".format(dataset))

        elif mode == "predict":
            print("Loading parameters")
            try:
                df_x = pd.read_pickle("./{}/params.pkl".format(dataset))
                df_y = pd.read_pickle("./{}/y_params.pkl".format(dataset))
                hmm.set_params(df_x, df_y)
            except:
                print("Parameters file can't be found, make sure to run in train mode first, and make sure parameter file from part 2 exist")

            print("Running viterbi")
            hmm.viterbi2("./{}/dev.in".format(dataset), './{}/dev.p3.out'.format(dataset))

d = Set()
d.set_training('./SG/train')

hmm = HMM(d)
