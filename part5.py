import pandas as pd
import numpy as np
import math as m
import pickle
import sys


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

    # Ommitted: to estimate transition and emission params

    def set_params(self, dfx, dfy):
        self.emission_params = dfx
        self.transistion_params = dfy

    # From part 3: viterbi

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
        f_in = open(in_file, 'r')
        f_out = open(out_file, 'w')
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
                    b_probs = self.max_b(words[j + 1])
                    ab = []
                    for k, l in zip(a_probs, b_probs):
                        ab.append(k * l)
                    pi_j = pi_j * max(ab)
                    pos = np.argmax(ab)
                    new_tag = self.tags[pos]
                    newtags_list.append(new_tag)

                    some_tag = new_tag
                    scores.append(pi_j)

                    if words[j + 2] == '\n':
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

    ########## PART 4 ##########
    def set_word_list(self, file_in):
        # word list will be a list of list
        f = open(file_in, "r", encoding="utf-8")
        content = f.read()

        words = []
        w = []

        for data in content.split("\n"):
            if data == "":
                if w != []:
                    words.append(w)
                    w = []
            else:
                w.append(data)

        # words = np.array(words)
        # df = pd.DataFrame({'words': words}, columns=['words'])
        # self.words = df.words.unique()
        self.words = words

    def count_y(self, y):
        df = self.training_set
        df = df[df['tags'] == y]
        count = df.shape[0]
        return count

    def convert_param(self):
        # Convert DataFrame to Dictionary
        T = ['B-NP', 'I-NP', 'B-VP', 'B-ADVP', 'B-ADJP', 'I-ADJP', 'B-PP', 'O', 'S', 'B-SBAR', 'I-VP', 'I-ADVP',
             'B-PRT', 'I-PP', 'B-CONJP', 'I-CONJP', 'B-INTJ', 'I-INTJ', 'I-SBAR', 'B-UCP', 'I-UCP', 'B-LST']
        # T = ['O', 'B-neutral', 'I-neutral', 'S', 'B-positive', 'I-positive', 'B-negative', 'I-negative']

        # Convert transition param
        t_param_dic = {}
        for tag_from in T:
            for tag_to in T:
                key = (tag_from, tag_to)
                row = self.transistion_params.loc[self.transistion_params['tags'] == tag_from]
                value = row['y_params'].values[0][T.index(tag_to)]
                t_param_dic[key] = value

        self.tr_param_dic = t_param_dic

        # Convert emmision param
        e_param_dic = {}
        words = self.emission_params['words'].values
        for w in words:
            # to skip 'S'
            T_temp = ['B-NP', 'I-NP', 'B-VP', 'B-ADVP', 'B-ADJP', 'I-ADJP', 'B-PP', 'O', 'B-SBAR', 'I-VP', 'I-ADVP',
                      'B-PRT', 'I-PP', 'B-CONJP', 'I-CONJP', 'B-INTJ', 'I-INTJ', 'I-SBAR', 'B-UCP', 'I-UCP', 'B-LST']
            # T_temp = ['O', 'B-neutral', 'I-neutral', 'B-positive', 'I-positive', 'B-negative', 'I-negative']
            for tag in T_temp:
                key = (w, tag)
                row = self.emission_params.loc[self.emission_params['words'] == w]
                value = row['params'].values[0][T_temp.index(tag)]
                e_param_dic[key] = value

        self.em_param_dic = e_param_dic

    def default_param(self):
        T = ['B-NP', 'I-NP', 'B-VP', 'B-ADVP', 'B-ADJP', 'I-ADJP', 'B-PP', 'O', 'S', 'B-SBAR', 'I-VP', 'I-ADVP',
             'B-PRT', 'I-PP', 'B-CONJP', 'I-CONJP', 'B-INTJ', 'I-INTJ', 'I-SBAR', 'B-UCP', 'I-UCP', 'B-LST']
        # T = ['O', 'B-neutral', 'I-neutral', 'S', 'B-positive', 'I-positive', 'B-negative', 'I-negative']
        default = {}
        for tag in T:
            default[tag] = 0.5 / float(self.count_y(tag) + 0.5)
        return default

    def viterbi_kbest(self, k=3, select=-1):
        if select==-1:
            select = k-1
        
        T = ['O', 'B-neutral', 'I-neutral', 'S', 'B-positive', 'I-positive', 'B-negative', 'I-negative']
        T_dict = {0:'O', 1:'B-neutral', 2:'I-neutral', 3:'S', 4:'B-positive', 5:'I-positive', 6:'B-negative', 7:'I-negative'}

        if self.dataset == "EN":
            T = ['B-NP', 'I-NP', 'B-VP', 'B-ADVP', 'B-ADJP', 'I-ADJP', 'B-PP', 'O', 'S', 'B-SBAR', 'I-VP', 'I-ADVP', 'B-PRT', 'I-PP', 'B-CONJP', 'I-CONJP', 'B-INTJ', 'I-INTJ', 'I-SBAR', 'B-UCP', 'I-UCP', 'B-LST']
            T_dict = {0: 'B-NP', 1: 'I-NP', 2: 'B-VP', 3: 'B-ADVP', 4: 'B-ADJP', 5: 'I-ADJP', 6: 'B-PP', 7: 'O', 8: 'S', 9: 'B-SBAR', 10: 'I-VP', 11: 'I-ADVP', 12: 'B-PRT', 13: 'I-PP', 14: 'B-CONJP', 15: 'I-CONJP', 16: 'B-INTJ', 17: 'I-INTJ', 18: 'I-SBAR', 19: 'B-UCP', 20: 'I-UCP', 21: 'B-LST'}

        y_pred = []
        default = self.default_param()

        for x in self.words:
            y = []
            temp = []
            temp_f = []

            # Base
            for s in range(len(T)):
                k_em = (x[0], T[s])
                k_tr = ('S', T[s])

                if k_em in self.em_param_dic:
                    # if any of the param is 0, log(0) is undefined, so set to a negative score
                    if self.em_param_dic[k_em] == 0 or self.tr_param_dic[k_tr] == 0:
                        score = -1000000000
                    else:
                        score = m.log(1.0 * self.em_param_dic[k_em] * self.tr_param_dic[k_tr])

                # If k_em doesnt exist in the dictionary asa key, use default param
                else:
                    if self.tr_param_dic[k_tr] != 0:
                        score = m.log(1.0 * default[T[s]] * self.tr_param_dic[k_tr])
                    else:
                        score = -1000000000

                temp.append(['S', s, 0, score])
                n = ['S', s, 0, -1000000000]
                for i in range(k - 1):
                    temp.append(n)

                temp_f.append(temp)
                temp = []

            y.append(temp_f)
            temp = []

            # Forward
            for i in range(len(x) - 1):
                i = i + 1
                for s_to in range(len(T)):
                    paths = []
                    scores = []

                    for s_fr in range(len(T)):
                        k_em = (x[i], T[s_to])
                        k_tr = (T[s_fr], T[s_to])
                        if len(y[i - 1][s_fr]) == 1:
                            if k_em not in self.em_param_dic:
                                if self.tr_param_dic[k_tr] != 0:
                                    score = float(y[i - 1][s_fr][0][3]) + m.log(float(default[T[s_to]])) + m.log(
                                        float(self.tr_param_dic[k_tr]))
                                else:
                                    score = -100000000
                            else:
                                if self.em_param_dic[k_em] == 0 or self.tr_param_dic[k_tr] == 0:
                                    score = float(y[i - 1][s_fr][0][3]) + float(-10000000)
                                else:
                                    score = float(y[i - 1][s_fr][0][3]) + m.log(float(self.em_param_dic[k_em])) + m.log(
                                        float(self.tr_param_dic[k_tr]))
                            paths.append([s_fr, s_to, 0, score])

                        else:
                            for e in range(len(y[i - 1][s_fr])):
                                if k_em not in self.em_param_dic:
                                    if self.tr_param_dic[k_tr] != 0:
                                        score = float(y[i - 1][s_fr][e][3]) + m.log(float(default[T[s_to]])) + m.log(
                                            float(self.tr_param_dic[k_tr]))
                                    else:
                                        score = -1000000000
                                else:
                                    if self.em_param_dic[k_em] == 0 or self.tr_param_dic[k_tr] == 0:
                                        score = float(y[i - 1][s_fr][e][3]) + float(-10000000)
                                    else:
                                        score = float(y[i - 1][s_fr][e][3]) + m.log(
                                            float(self.em_param_dic[k_em])) + m.log(float(self.tr_param_dic[k_tr]))
                                paths.append([s_fr, s_to, e, score])

                    # sort top 3
                    top_three = sorted(paths, key=lambda top_three: top_three[3], reverse=True)[:k]
                    temp.append(top_three)

                y.append(temp)
                temp = []

            # Final
            temp = []
            paths = []
            top_three = []
            for s_fr in range(len(T)):
                for e in range(k):
                    layer_f = len(x)
                    if self.tr_param_dic[(T[s_fr], 'S')] != 0:
                        if layer_f == 1:
                            score = float(y[layer_f - 1][s_fr][0][3]) + m.log(float(self.tr_param_dic[(T[s_fr], 'S')]))
                        else:
                            score = float(y[layer_f - 1][s_fr][e][3]) + m.log(float(self.tr_param_dic[(T[s_fr], 'S')]))
                    paths.append([s_fr, 'S', e, score])

            # sort top 3
            top_three = sorted(paths, key=lambda top_three: top_three[3], reverse=True)[:k]

            temp.append(top_three)
            y.append(temp)

            # Backtrack
            # print(top_three)
            y1 = top_three[select][0]
            j = top_three[select][2]
            y_pred_num = []
            for i in range(len(x) - 1, 0, -1):
                y2 = y1
                y_pred_num.append(y2)
                y1 = y[i][y2][j][0]
                j = y[i][y2][j][2]

            y_pred_num.append(y1)
            y_pred_label = []

            for i in range(len(y_pred_num) -1, -1, -1):
                y = T_dict[y_pred_num[i]]
                y_pred_label.append(y)
            y_pred.append(y_pred_label)
        
        return y_pred

    def part_5(self, k=3, file_name="./dev.p5.out"):
        y = []
        y_predict = []
        weights = []
        for i in range(k):
            weights.append((40 - i) ** 2 * 0.5)

        for select in range(k):
            print(select)
            y.append(self.viterbi_kbest(k, select))

        for m in range(len(y[0])):
            ym_predict = []
            for i in range(len(y[0][m])):
                temp = {}
                for j in range(k):
                    if y[j][m][i] not in temp:
                        temp[y[j][m][i]] = 0
                    temp[y[j][m][i]] = temp[y[j][m][i]] + weights[j]
                data = sorted(temp.items(), key=lambda x: x[1], reverse=True)
                most = data[0][0]
                ym_predict.append(most)
            y_predict.append(ym_predict)


        # print("writing file")
        with open(file_name, "w", encoding="utf-8") as f_out:
            for i in range(len(self.words)):
                x_i = self.words[i]
                y_i = y_predict[i]
                for j in range(len(x_i)):
                    f_out.write("{} {}\n".format(x_i[j], y_i[j]))
                f_out.write("\n")

        return y_predict


if __name__=="__main__":
    if len(sys.argv) < 3:
        print("Make sure at least python 3.8 is installed")
        print("Run the file in this format")
        print("python part5.py [dataset] [mode]")
        print("dataset can be EN,SG,CN") # sys.argv[1]
        print("mode can be train or predict") # sys.argv[2]

    else:
        dataset = sys.argv[1]
        mode = sys.argv[2]

        d = Set()
        d.set_training('./{}/train'.format(dataset))
        hmm = HMM(d)
        hmm.dataset = dataset

        # 1. prepare x /EN/dev.in
        print("Setting word list")
        hmm.set_word_list("./{}/dev.in".format(dataset))
        
        # 2. em param and trans param
        print("Loading emission and transition parameters")
        try:
            dfx = pd.read_pickle("./{}/params.pkl".format(dataset))
            dfy = pd.read_pickle("./{}/y_params.pkl".format(dataset))
            hmm.set_params(dfx, dfy)
        except:
            print("Parameters file not found, make sure to run part2.py and part3.py in train mode to generate the parameters file")

        if mode == "train":
            # 3. Transform em and tr param for ease of access
            print("Creating dictionary of parameters")
            hmm.convert_param()

            # 3.a Saving Params
            with open("./{}/em_dic.p".format(dataset), "wb") as fp:
                pickle.dump(hmm.em_param_dic, fp, protocol=pickle.HIGHEST_PROTOCOL)

            with open("./{}/tr_dic.p".format(dataset), "wb") as fp:
                pickle.dump(hmm.tr_param_dic, fp, protocol=pickle.HIGHEST_PROTOCOL)
        
        elif mode == "predict":
            # 3.b Loading params
            with open("./{}/em_dic.p".format(dataset), "rb") as fp:
                hmm.em_param_dic = pickle.load(fp)

            with open("./{}/tr_dic.p".format(dataset), "rb") as fp:
                hmm.tr_param_dic = pickle.load(fp)

            # 4. Do part 5
            print("Doing part 5 with k=3")
            filename = "./{}/dev.p5.out".format(dataset)
            y_pred_part5 = hmm.part_5(3, filename)
            print("Output is saved to {}".format(filename))
