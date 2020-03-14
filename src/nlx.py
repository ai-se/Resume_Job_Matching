from __future__ import print_function, division
import pandas as pd
import numpy as np
from pdb import set_trace
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from demos import cmd

try:
   import cPickle as pickle
except:
   import pickle


class JobResume():
    def __init__(self):
        self.job = []
        self.title = []
        self.indices = {}

    def load(self,path="../data/nlx/",name="machinist"):
        df = pd.read_csv(path+name+".csv")
        start = len(self.job)
        self.job+=[x for x in df["description"]]
        end = len(self.job)
        self.title+=[name]*len(df)
        self.indices[name] = range(start,end)



    def lda(self, seed=None, num_topics=10, alpha=0.1, eta=0.01, norm=None):
        if seed:
            np.random.seed(seed)
        import lda
        tfer = TfidfVectorizer(lowercase=True, stop_words="english", norm=None, use_idf=False,
                               decode_error="ignore")
        target = self.job

        self.csr_mat=tfer.fit_transform(target)
        lda1 = lda.LDA(n_topics=num_topics, alpha=alpha, eta=eta, n_iter=200)
        self.csr_mat = lda1.fit_transform(self.csr_mat.astype(int))

        vocab = tfer.vocabulary_
        self.vocab =  np.array(vocab.keys())[np.argsort(vocab.values())]
        self.topic_words = lda1.topic_word_
        n_topic_words = 8
        for i,topic_dist in enumerate(self.topic_words):
            topic = self.vocab[np.argsort(topic_dist)[-n_topic_words:][::-1]]
            print('Topic {}: {}'.format(i,' '.join(topic)))
        if norm:
            self.csr_mat = csr_matrix(preprocessing.normalize(self.csr_mat,norm=norm,axis=1))
        return

    def feature_distribution(self):
        dis = {}
        for name in self.indices:
            body = self.csr_mat[self.indices[name]]
            mean = sum(body)/float(len(body))
            dis[name] = mean
        return dis



def lda_distribution():
    x = JobResume()     # Load data
    names = ["machine_operator","machinist"]
    for name in names:
        x.load(name=name)
    x.lda(seed=0,num_topics=10,norm=None)
    dis = x.feature_distribution()
    df = pd.DataFrame(dis,columns=names)
    df.to_csv("../figure/nlx_distribution.csv")


if __name__ == "__main__":
    eval(cmd())

