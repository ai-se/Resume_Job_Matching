from __future__ import print_function, division
import pandas as pd
import numpy as np
from pdb import set_trace
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing

from demos import cmd

import re


class JobResume():
    def __init__(self,jobfile = "../data/job.csv",resumefile="../data/resume.csv"):
        self.jobs = pd.read_csv(jobfile)
        self.resumes = pd.read_csv(resumefile)

    def prepare(self):
        resumes = [x for x in self.resumes["Resume"]]
        jobs = [x for x in self.jobs["jobpost"]]
        self.num_resume = len(resumes)
        self.num_job = len(jobs)
        self.content = resumes+jobs

    def lda(self):
        import lda
        tfer = TfidfVectorizer(lowercase=True, stop_words="english", norm=None, use_idf=False,
                               decode_error="ignore")
        self.csr_mat = tfer.fit_transform(self.content)

        lda1 = lda.LDA(n_topics=100, alpha=0.1, eta=0.01, n_iter=200)
        self.csr_mat = lda1.fit_transform(self.csr_mat.astype(int))
        self.csr_mat = preprocessing.normalize(self.csr_mat,norm='l2',axis=1)
        return

    def doc2vec(self):
        from gensim.models import Doc2Vec
        from gensim.models.doc2vec import TaggedDocument
        import multiprocessing

        def convert_sentences(sentence_list):
            for i in range(len(sentence_list)):
                for char in ['.', ',', '!', '?', ';', ':']:
                    sentence_list[i] = sentence_list[i].replace(char, ' ' + char + ' ')
            return [TaggedDocument(words=sentence_list[i].split(), tags=[i]) for i in range(len(sentence_list))]

        def normalize(x, p=2):
            xx = np.linalg.norm(x, p)
            return x / xx if xx else x



        content1 = convert_sentences(self.content)
        model = Doc2Vec(size=300, window=10, min_count=5, workers=multiprocessing.cpu_count(),alpha=0.025, min_alpha=0.025)
        model.build_vocab(content1)

        for epoch in range(10):
            model.train(content1, total_examples=model.corpus_count, epochs=model.iter)
            model.alpha -= 0.002  # decrease the learning rate
            model.min_alpha = model.alpha  # fix the learning rate, no decay

        self.csr_mat = np.array([normalize(model.infer_vector(x.words, alpha=model.alpha, min_alpha=model.min_alpha),p=2) for x in content1])
        return

    def tfidf(self):
        tfidfer = TfidfVectorizer(lowercase=True, stop_words="english", norm=None, use_idf=True, smooth_idf=False,
                                sublinear_tf=False,decode_error="ignore",max_features=4000)
        tfidfer.fit(self.content)
        self.voc = tfidfer.vocabulary_.keys()

        ##############################################################

        ### Term frequency as feature, L2 normalization ##########
        tfer = TfidfVectorizer(lowercase=True, stop_words="english", norm=u'l2', use_idf=False,
                        vocabulary=self.voc,decode_error="ignore")
        # tfer = TfidfVectorizer(lowercase=True, stop_words="english", norm=None, use_idf=False,
        #                 vocabulary=self.voc,decode_error="ignore")
        self.csr_mat=tfer.fit_transform(self.content)
        return


if __name__ == "__main__":
    eval(cmd())