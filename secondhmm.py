import pandas as pd
import numpy as np
import math as m
import pickle


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

    # Emission params & default params i think no need
    def count_y(self, y):
        df = self.training_set
        df = df[df['tags'] == y]
        count = df.shape[0]
        return count

    def default_param(self):
        T = ['B-NP', 'I-NP', 'B-VP', 'B-ADVP', 'B-ADJP', 'I-ADJP', 'B-PP', 'O', 'S', 'B-SBAR', 'I-VP', 'I-ADVP', 'B-PRT', 'I-PP', 'B-CONJP', 'I-CONJP', 'B-INTJ', 'I-INTJ', 'I-SBAR', 'B-UCP', 'I-UCP', 'B-LST']
        # T = ['O', 'B-neutral', 'I-neutral', 'S', 'B-positive', 'I-positive', 'B-negative', 'I-negative']
        default = {}
        for tag in T:
            default[tag] = 0.5/float(self.count_y(tag) + 0.5)
        return default

    # To estimate the transition parameters
    def count_y_to_y(self, prev_y1, prev_y2, y):
        # prev_y1 = yi-1
        # prev_y2 = yi-2
        df = self.training_set
        count = 0
        for i in range(len(df['tags'])-2):
            if df['tags'][i] == prev_y2 and df['tags'][i+1] == prev_y1 and df['tags'][i+2] == y:
                count += 1
        return count

    def count_prev_y(self, prev_y, y):
        df = self.training_set
        count = 0
        for i in range(len(df['tags'])-1):
            if df['tags'][i] == prev_y and df['tags'][i+1]:
                count += 1
        return count

    def trans_params(self, prev_y1, prev_y2, y):
        num = self.count_y_to_y(prev_y1, prev_y2, y)
        den = self.count_prev_y(prev_y1, y)
        q = num / den
        return q

    # I'm lazy to figure out as for now so imma make it a dictionary (y-2, y-1, y)
    def train_trans_params(self):
        trans_param = {}
    
        print('training...')
        for tag_2 in self.tags:
            for tag_1 in self.tags:
                for tag in self.tags:
                    key = (tag_2, tag_1, tag)
                    value = self.trans_params(tag_2, tag_1, tag)
                    trans_param[key] = value

        self.trans_params = trans_param

    def set_params(self, dfx, dfy):
        self.emission_params = dfx
        self.transistion_params = dfy

    # From part 3: viterbi2
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

    ############### PART 5 ###############
    def convert_param(self):
        # Convert DataFrame to Dictionary
        T = ['B-NP', 'I-NP', 'B-VP', 'B-ADVP', 'B-ADJP', 'I-ADJP', 'B-PP', 'O', 'S', 'B-SBAR', 'I-VP', 'I-ADVP', 'B-PRT', 'I-PP', 'B-CONJP', 'I-CONJP', 'B-INTJ', 'I-INTJ', 'I-SBAR', 'B-UCP', 'I-UCP', 'B-LST']
        # T = ['O', 'B-neutral', 'I-neutral', 'S', 'B-positive', 'I-positive', 'B-negative', 'I-negative']

        # I already calculated it and saved in this format so no need
        # Convert transition param
        # t_param_dic = {}
        # for tag_from in T:
        #     for tag_to in T:
        #         key = (tag_from, tag_to)
        #         row = self.transistion_params.loc[self.transistion_params['tags'] == tag_from]
        #         value = row['y_params'].values[0][T.index(tag_to)]
        #         t_param_dic[key] = value

        # self.tr_param_dic = t_param_dic

        # Convert emmision param
        e_param_dic = {}
        words = self.emission_params['words'].values
        for w in words:
            # to skip 'S'
            T_temp = ['B-NP', 'I-NP', 'B-VP', 'B-ADVP', 'B-ADJP', 'I-ADJP', 'B-PP', 'O', 'B-SBAR', 'I-VP', 'I-ADVP', 'B-PRT', 'I-PP', 'B-CONJP', 'I-CONJP', 'B-INTJ', 'I-INTJ', 'I-SBAR', 'B-UCP', 'I-UCP', 'B-LST']
            # T_temp = ['O', 'B-neutral', 'I-neutral', 'B-positive', 'I-positive', 'B-negative', 'I-negative']
            for tag in T_temp:
                key = (w, tag)
                row = self.emission_params.loc[self.emission_params['words'] == w]
                value = row['params'].values[0][T_temp.index(tag)]
                e_param_dic[key] = value

        self.em_param_dic = e_param_dic
    
    # Second order hmm
    def second_order(self, in_file, out_file):
        sentences = self.get_sentence(in_file)
        f_out = open(out_file, "w")
        T_temp = ['B-NP', 'I-NP', 'B-VP', 'B-ADVP', 'B-ADJP', 'I-ADJP', 'B-PP', 'O', 'S', 'B-SBAR', 'I-VP', 'I-ADVP', 'B-PRT', 'I-PP', 'B-CONJP', 'I-CONJP', 'B-INTJ', 'I-INTJ', 'I-SBAR', 'B-UCP', 'I-UCP', 'B-LST']
        y_pred = []

        for sentence in sentences:
            y = []
            phi = {}
            phi_args = {}
            n = len(sentence)
            # print(n)
            
            # Base:
            # phi(-1,'S') and phi(0,'S') = 1, (-1,T[s]) and (0,T[s]) = 0
            for s in range(len(T_temp)):
                
                k_a = (-1, T_temp[s])
                k_b = (0, T_temp[s])

                if T_temp[s] == 'S':
                    phi[k_a] = 1
                    phi[k_b] = 1
                else:
                    phi[k_a] = 0
                    phi[k_b] = 0

            T = ['B-NP', 'I-NP', 'B-VP', 'B-ADVP', 'B-ADJP', 'I-ADJP', 'B-PP', 'O', 'B-SBAR', 'I-VP', 'I-ADVP', 'B-PRT', 'I-PP', 'B-CONJP', 'I-CONJP', 'B-INTJ', 'I-INTJ', 'I-SBAR', 'B-UCP', 'I-UCP', 'B-LST']

            # "Base Case" of k=1
            phi_base = {}
            if sentence[1] not in self.words:
                x = '#UNK#'
                for w in range(len(T)):
                    key = (1,T[w])
                    phi_base[key] = self.prev_trans_params[('S',T[w])] * self.em_params[(x, T[w])]
                    phi[key] = phi_base[key]
                    phi_args[key] = (key[0], key[1], T[w])
            else:
                for w in range(len(T)):
                    key = (1,T[w])
                    phi_base[key] = self.prev_trans_params[('S',T[w])] * self.em_params[(sentence[1], T[w])]
                    phi[key] = phi_base[key]
                    phi_args[key] = (key[0], key[1], T[w])
            # print(phi)
            # Get max of this base

            # Forward recursive
            for k in range(2, n-2):
                if sentence[k] not in self.words:
                    x = '#UNK#'
                else:
                    x = sentence[k]

                for w in range(len(T)):
                    # key will be (u,v), value is phi*a, uv is the tag (T[x])
                    phi_a = {}
                    for u in range(len(T)):
                        for v in range(len(T)):
                            phi_value = phi[(k-1,T[v])]
                            if phi_value != 0:
                                # print(phi_value)
                                a_value = self.trans_params[(T[u], T[v], T[w])]
                                phi_a[(T[u],T[v])] = phi_value * a_value
                            else:
                                phi_a[(T[u],T[v])] = 0
                    # get max value in the phi_a
                    # print(phi_a)
                    max_key = max(phi_a, key=phi_a.get)
                    max_value = phi_a[max_key]
                    phi[(k,T[w])] = max_value * self.em_params[(x, T[w])]
                    phi_args[(k,T[w])] = (max_key[0], max_key[1], T[w])

            # Final Transition
            # Transiton from n to 'S' (STOP)
            final_phi_a = {}
            for v in range(len(T)):
                for w in range(len(T)):
                    phi_value = phi[(n-3,T[w])]
                    a_value = self.trans_params[(T[v], T[w], 'S')]
                    final_phi_a[(T[v],T[w])] = phi_value * a_value
            f_max_key = max(final_phi_a, key=final_phi_a.get)
            f_max_value = final_phi_a[f_max_key]
            phi[(n-2, 'S')] = f_max_value
            phi_args[(n-2, 'S')] = (f_max_key[0], f_max_key[1], 'S')

            # Back tracking
            y.append('S')
            # print(phi_args)
            for i in range(n-2, 0,-1):
                best_route = phi_args[(i,y[0])]
                y.insert(0, best_route[1])
                # print(y)
            y_pred.append(y)
            # remove last s
            # print(y)
            del y[-1]
            # this break so that i just see the result of the first sentence for now
            
            # element 0 and last element of sentence should be "forgotten"
            x = sentence[1:len(sentence)-1]
            for i in range(len(y)):
                f_out.write("{} {}\n".format(x[i], y[i]))
            f_out.write("\n")

        f_out.close()
        return y_pred
        


if __name__=="__main__":
    d = Set()
    d.set_training('./EN/train')
    hmm = HMM(d)

    # print(hmm.training_set)

    # Generating trans params and saving
    # hmm.train_trans_params()
    # with open("./EN/trans_param_second.p", "wb") as fp:
    #     pickle.dump(hmm.trans_params, fp, protocol=pickle.HIGHEST_PROTOCOL)

    # Transform em param and saving
    # print("Creating dictionary of em params")
    # hmm.convert_param()
    # with open("./EN/em_param_second.p", "wb") as fp:
    #     pickle.dump(hmm.em_param_dic, fp, protocol=pickle.HIGHEST_PROTOCOL)


    # Loading trans params
    print("loading trans params")
    with open("./EN/trans_param_second.p", "rb") as fp:
        hmm.trans_params = pickle.load(fp)
    with open("./EN/tr_dic.p", "rb") as fp:
        hmm.prev_trans_params = pickle.load(fp)

    # Loading emm params (since it's the same)
    print("loading em params")
    with open("./EN/em_param_second.p", "rb") as fp:
        hmm.em_params = pickle.load(fp)

    print("running second order")
    y_predict = hmm.second_order("./EN/dev.in", "./EN/second_order_test.out")