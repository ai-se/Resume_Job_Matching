from __future__ import print_function, division
import pandas as pd
import numpy as np
from pdb import set_trace
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import spearmanr
from demos import cmd

import re


class JobResume():
    def __init__(self,jobfile = "../data/job.csv",resumefile="../data/resume.csv"):
        self.jobs = pd.read_csv(jobfile)
        self.resumes = pd.read_csv(resumefile)

    def print_job(self,job_id):
        print("Job Post:")
        print(self.jobs["jobpost"][job_id])

    def print_resume(self,resume_id):
        print("Resume:")
        print(self.resumes["Resume"][resume_id].decode('string_escape'))

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
        self.csr_mat = csr_matrix(preprocessing.normalize(self.csr_mat,norm='l2',axis=1))
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
        model = Doc2Vec(vector_size=300, window=10, min_count=5, workers=multiprocessing.cpu_count(),alpha=0.025, min_alpha=0.025)
        model.build_vocab(content1)

        for epoch in range(10):
            model.train(content1, total_examples=model.corpus_count, epochs=model.iter)
            model.alpha -= 0.002  # decrease the learning rate
            model.min_alpha = model.alpha  # fix the learning rate, no decay

        self.csr_mat = csr_matrix([normalize(model.infer_vector(x.words, alpha=model.alpha, min_alpha=model.min_alpha),p=2) for x in content1])
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

    def match_resume(self,resume_id,num):
        target = self.csr_mat[resume_id].transpose()
        jobs = self.csr_mat[self.num_resume:]
        probs = (jobs*target).toarray().flatten()
        order = np.argsort(probs)[::-1][:num]
        return order, probs[order]

    def match_job(self,job_id,num):
        target = self.csr_mat[self.num_resume+job_id].transpose()
        jobs = self.csr_mat[:self.num_resume]
        probs = (jobs*target).toarray().flatten()
        order = np.argsort(probs)[::-1][:num]
        return order, probs[order]

    def dimensionality_reduction(self,n=3):
        pca = PCA(n_components=n)
        self.reduced_mat = pca.fit_transform(self.csr_mat.toarray())

    def clustering(self, n=5):
        self.clusters = KMeans(n_clusters=n, random_state=0).fit(self.csr_mat).labels_

    def visualization(self,name=''):
        self.dimensionality_reduction()
        self.clustering(n=5)

        font = {'family': 'cursive',
            'weight': 'bold',
            'size': 20}


        plt.rc('font', **font)
        paras = {'lines.linewidth': 4, 'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,
                 'figure.autolayout': True, 'figure.figsize': (16, 6)}
        plt.rcParams.update(paras)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        resumes = {}
        jobs = {}
        for cat in set(self.clusters):
            jobs[cat] = {"x": [], "y": [], "z":[]}
            resumes[cat] = {"x":[], "y":[], "z":[]}
        for i,row in enumerate(self.reduced_mat):
            cat = self.clusters[i]
            if i < self.num_resume:
                resumes[cat]["x"].append(row[0])
                resumes[cat]["y"].append(row[1])
                resumes[cat]["z"].append(row[2])
            else:
                jobs[cat]["x"].append(row[0])
                jobs[cat]["y"].append(row[1])
                jobs[cat]["z"].append(row[2])
        colors = ["red","blue","green","gray", "orange"]
        for cat in set(self.clusters):
            ax.scatter(jobs[cat]["x"],jobs[cat]["y"],jobs[cat]["z"],marker ="o",color=colors[cat])
            ax.scatter(resumes[cat]["x"],resumes[cat]["y"],resumes[cat]["z"], marker ="^",color=colors[cat])
        plt.savefig("../figure/"+name+"visualization3D.png")


def test():
    x = JobResume()     # Load data
    x.prepare()         # Preprocessing
    x.tfidf()         # Encode every resume and job post
    x.visualization()
    set_trace()

    # Find top 5 most similar resumes to Job post ID 0.
    matched_resumes, probs = x.match_job(0,5)
    # Print the ID of the recommended top 5 resumes and their cosine distance to the target job post.
    for i,r in enumerate(matched_resumes):
        print("ID: %d, Prob: %f" %(r,probs[i]))

    set_trace()

    # Find top 5 most similar job posts to Resume ID 0.
    matched_jobs, probs = x.match_resume(0, 5)
    # Print the ID of the recommended top 5 job posts and their cosine distance to the target resume.
    for i, r in enumerate(matched_jobs):
        print("ID: %d, Prob: %f" % (r, probs[i]))

    set_trace()

    x.print_resume(matched_resumes[0])

def consist():
    x = JobResume()     # Load data
    x.prepare()         # Preprocessing
    x.doc2vec()         # Encode every resume and job post
    y = JobResume()     # Load data
    y.prepare()         # Preprocessing
    y.lda()         # Encode every resume and job post
    z = JobResume()     # Load data
    z.prepare()         # Preprocessing
    z.tfidf()         # Encode every resume and job post
    matched_resumes_x, probs_x = x.match_job(0,5)
    matched_resumes_y, probs_y = y.match_job(0,5)
    matched_resumes_z, probs_z = z.match_job(0,5)
    spearmanr(probs_x,probs_y)
    spearmanr(probs_z,probs_y)
    spearmanr(probs_x,probs_z)

    matched_jobs_x, probs_x = x.match_resume(0, 5)
    matched_jobs_y, probs_y = y.match_resume(0, 5)
    matched_jobs_z, probs_z = z.match_resume(0, 5)
    spearmanr(probs_x,probs_y)
    spearmanr(probs_z,probs_y)
    spearmanr(probs_x,probs_z)

    x.visualization(name="doc2vec_")
    y.visualization(name="lda_")
    z.visualization(name="tfidf_")
    set_trace()

if __name__ == "__main__":
    eval(cmd())