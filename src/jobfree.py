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
import random
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import spearmanr
from demos import cmd
from collections import Counter

try:
   import cPickle as pickle
except:
   import pickle

import re


class JobResume():
    def __init__(self,jobfile = "../data/combinedJob_V2.csv",resumefile="../data/combinedResume_V2.csv"):
        self.jobs = pd.read_csv(jobfile)
        self.resumes = pd.read_csv(resumefile)
        self.resumes["Text"] = [x.decode('string_escape') for x in self.resumes["Text"]]
        self.jobs["Text"] = [x.decode('string_escape') for x in self.jobs["Text"]]

        self.resumes["categories"] = [set(x.split(",")) for x in self.resumes["Job Title"]]
        self.jobs["categories"] = [set(x.split(",")) for x in self.jobs["Job Title"]]

    def prepare(self):
        self.resume_info = [x for x in self.resumes["Text"]]
        self.job_post = [x for x in self.jobs["Text"]]
        self.num_resume = len(self.resume_info)
        self.num_job = len(self.job_post)
        self.content = self.resume_info+self.job_post

    def print_job(self,job_id):
        print("Job Post:")
        print(self.job_post[job_id])

    def print_resume(self,resume_id):
        print("Resume:")
        print(self.resume_info[resume_id])

    def lda(self, content="all", seed=None, num_topics=100, alpha=0.1, eta=0.01, norm='l2'):
        if seed:
            np.random.seed(seed)
        import lda
        tfer = TfidfVectorizer(lowercase=True, stop_words="english", norm=None, use_idf=False,
                               decode_error="ignore")
        if content == "resume":
            target = self.resume_info
        elif content == "job":
            target = self.job_post
        else:
            target = self.content
        # tfidfer = TfidfVectorizer(lowercase=True, stop_words="english", norm=None, use_idf=True, smooth_idf=False,
        #                         sublinear_tf=False,decode_error="ignore",max_features=4000)
        # tfidfer.fit(target)
        # voc = tfidfer.vocabulary_.keys()
        # remove_words = ["job","application","armenia"]
        # for word in remove_words:
        #     if word in voc:
        #         voc.remove(word)

        # tfer = TfidfVectorizer(lowercase=True, stop_words="english", norm=None, use_idf=False,
        #                 vocabulary=voc,decode_error="ignore")
        # tfer = TfidfVectorizer(lowercase=True, stop_words="english", norm=None, use_idf=False,
        #                 vocabulary=self.voc,decode_error="ignore")
        self.csr_mat=tfer.fit_transform(target)
        lda1 = lda.LDA(n_topics=num_topics, alpha=alpha, eta=eta, n_iter=200)
        self.csr_mat = lda1.fit_transform(self.csr_mat.astype(int))
        self.classes = np.argmax(self.csr_mat,axis=1)
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

    def doc2vec(self):

        from gensim.models import Doc2Vec
        from gensim.models.doc2vec import TaggedDocument
        import multiprocessing
        np.random.seed(0)

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


    def match_resume(self,resume_id,num,subset=None):
        if subset==None:
            subset = range(self.num_job)
        target = self.csr_mat[resume_id].transpose()
        cans = self.csr_mat[self.num_resume:][subset]
        probs = (cans*target).toarray().flatten()
        order = np.argsort(probs)[::-1][:num]
        return np.array(subset)[order], probs[order]

    def match_job(self,job_id,num,subset=None):
        if subset==None:
            subset = range(self.num_resume)
        target = self.csr_mat[self.num_resume+job_id].transpose()
        cans = self.csr_mat[:self.num_resume][subset]
        probs = (cans*target).toarray().flatten()
        order = np.argsort(probs)[::-1][:num]
        return np.array(subset)[order], probs[order]

    def cos_dist(self,a,b):
        return (self.csr_mat[a]*(self.csr_mat[b].transpose())).toarray()[0]

    def dimensionality_reduction(self,n=3):
        pca = PCA(n_components=n)
        self.reduced_mat = pca.fit_transform(self.csr_mat.toarray())

    def clustering(self, n=5):
        np.random.seed(0)
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

    def hierarchy(self, n=10):
        classes = [{title: np.where(self.jobs["Title"]==title)[0].tolist()} for title in set(self.jobs["Title"]) if not pd.isnull(title)]
        m=self.csr_mat.shape[1]
        while len(classes)>n:
            print(len(classes))
            centers = []
            for dict in classes:
                indices = [item for sublist in dict.values() for item in sublist]
                try:
                    center = preprocessing.normalize(np.array([np.mean(self.csr_mat[indices].toarray(),axis=0)]),norm='l2',axis=1)[0]
                except:
                    set_trace()
                centers.append(center)
            centers = csr_matrix(np.array(centers))
            mat = (centers * centers.transpose()).toarray()
            for i in xrange(len(mat)):
                mat[i][i] = 0
            closest_i, closest_j = np.unravel_index(np.argmax(mat, axis=None), mat.shape)
            if closest_i > closest_j:
                closest_i, closest_j = closest_j, closest_i
            to_merge = classes.pop(closest_j)
            for title in to_merge:
                classes[closest_i][title] = to_merge[title]

        self.classes = range(len(self.jobs))
        for i,dict in enumerate(classes):
            for index in [item for sublist in dict.values() for item in sublist]:
                self.classes[index] = i


        new_jobs = []
        for job_title in classes:
            tmp = {title:len(job_title[title]) for title in job_title}
            new_jobs.append(np.array(tmp.keys())[np.argsort(tmp.values())[::-1][:8]])
        print(new_jobs)

    def gen_triplet(self, type="job"):

        def is_close(job_id, resume_id):
            return len(self.jobs["categories"][job_id] & self.resumes["categories"][resume_id]) > 0

        if type == "job":
            job_id = np.random.randint(self.num_job)
            resume_id_a = np.random.randint(self.num_resume)
            if is_close(job_id, resume_id_a):
                while True:
                    resume_id_b = np.random.randint(self.num_resume)
                    if not is_close(job_id, resume_id_b):
                        break
            else:
                resume_id_b = resume_id_a
                while True:
                    resume_id_a = np.random.randint(self.num_resume)
                    if is_close(job_id, resume_id_a):
                        break
            return (job_id, resume_id_a, resume_id_b)
        else:
            resume_id = np.random.randint(self.num_resume)
            job_id_a = np.random.randint(self.num_job)
            if is_close(job_id_a, resume_id):
                while True:
                    job_id_b = np.random.randint(self.num_job)
                    if not is_close(job_id_b, resume_id):
                        break
            else:
                job_id_b = job_id_a
                while True:
                    job_id_a = np.random.randint(self.num_job)
                    if is_close(job_id_a, resume_id):
                        break
            return (resume_id, job_id_a, job_id_b)

    def get_ids(self,category="Software Engineer",is_job=True):
        candidates = self.jobs["categories"] if is_job else self.resumes["categories"]
        return [i for i,can in enumerate(candidates)  if category in can]

    def feature_distribution(self):
        categories = set()
        for cans in self.jobs["categories"]:
            categories = categories | cans
        job_distribution = {}
        resume_distribution = {}
        for cat in categories:
            job_ids = self.get_ids(category=cat,is_job=True)
            resume_ids = self.get_ids(category=cat,is_job=False)
            jobs = self.csr_mat[np.array(job_ids)+self.num_resume]
            resumes = self.csr_mat[np.array(resume_ids)]
            job_distribution[cat] = (np.sum(jobs,axis=0)/len(jobs)).tolist()
            resume_distribution[cat] = (np.sum(resumes,axis=0)/len(resumes)).tolist()
        df_job = pd.DataFrame(job_distribution,columns=job_distribution.keys())
        df_resume = pd.DataFrame(resume_distribution,columns=job_distribution.keys())
        df_job.to_csv("../figure/job_distribution.csv")
        df_resume.to_csv("../figure/resume_distribution.csv")

    def gen_inclass_triplet(self,category="Software Engineer",is_target_job=True, num = 100):
        target_ids = self.get_ids(category, is_target_job)
        candidate_ids = self.get_ids(category, not is_target_job)
        targets = np.random.choice(target_ids,num, replace=False)
        triplets = []
        for target in targets:
            if is_target_job:
                cans, probs = self.match_job(target,2,candidate_ids)
            else:
                cans, probs = self.match_resume(target, 2, candidate_ids)
            triplets.append([target] + cans.tolist())
        # candidates = list(set(np.random.choice(candidate_ids,2,replace=False)))
        # triplet =  (np.random.choice(target_ids,1)[0], candidates[0], candidates[1])

        return triplets

def test():
    x = JobResume()     # Load data
    x.prepare()         # Preprocessing
    x.tfidf()         # Encode every resume and job post
    set_trace()

    x.visualization()

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




def triplet_test():
    margin = 0.0
    num_triplets = 10000
    score = 1/float(num_triplets)
    x = JobResume()     # Load data
    x.prepare()         # Preprocessing
    num_tries = 30
    result_job = {"tfidf":[],"lda":[],"doc2vec":[]}
    result_resume = {"tfidf":[],"lda":[],"doc2vec":[]}
    for treatment in [x.tfidf,x.lda]:
        name = treatment.__name__
        for tries in range(num_tries):
            treatment()
            r_job = 0
            r_resume = 0
            for epoch in range(num_triplets):
                triplet = x.gen_triplet(type="job")
                diff = x.cos_dist(triplet[0]+x.num_resume,triplet[1]) - x.cos_dist(triplet[0]+x.num_resume,triplet[2])
                if diff > margin:
                    r_job+=score
                elif diff < -margin:
                    r_job+=-score

                triplet = x.gen_triplet(type="resume")
                diff = x.cos_dist(triplet[0],triplet[1]+x.num_resume) - x.cos_dist(triplet[0],triplet[2]+x.num_resume)
                if diff > margin:
                    r_resume+=score
                elif diff < -margin:
                    r_resume+=-score
            result_job[name].append(r_job)
            result_resume[name].append(r_resume)
    print("targeting jobs")
    print(result_job)
    print("targeting resumes")
    print(result_resume)
    set_trace()



def trend():
    import matplotlib.patches as mpatches


    x = JobResume()     # Load data
    x.prepare()
    num_topics = 100
    x.lda(content="job",seed=5,num_topics=num_topics)
    print(Counter(x.classes))
    set_trace()
    representatives = np.array(np.argmax(x.csr_mat,axis=0))[0]
    print(representatives)
    set_trace()

    years = sorted(list(set(x.jobs["Year"])))
    result = []
    for year in years:
        indices = np.where(x.jobs["Year"]==year)[0]
        total = len(indices)
        tmp = Counter(x.classes[indices])
        row = {"Year": year}
        tmp_row = {}
        for i in xrange(num_topics):
            tmp_row[i]=float(tmp[i])/total
        row["order"] = [(key, tmp_row[key]) for key in np.array(tmp_row.keys())[np.argsort(tmp_row.values())]]
        result.append(row)

    COLORS_ALL = ["lightgray", "red", "blue", "darkslategray","yellow", "darkmagenta", "cyan", "saddlebrown","orange", "lime", "hotpink"]
    def get_color(index):
        return COLORS_ALL[index]

    width = 0.6
    plts = []
    x_axis = np.arange(0, len(years))
    y_offset = np.array([0] * len(years))
    colors_dict = {}
    top_topic_count = num_topics
    plt.figure(figsize=(8, 6))
    for index in range(top_topic_count):
        bar_val, color = [], []
        for i,row in enumerate(result):
            topic = row["order"][index]
            if topic[0] not in colors_dict:
                colors_dict[topic[0]] = get_color(topic[0])
            bar_val.append(topic[1])
            color.append(colors_dict[topic[0]])
        plts.append(plt.bar(x_axis, bar_val, width, color=color, bottom=y_offset))
        y_offset = np.add(y_offset, np.array(bar_val))
    plt.ylabel("Topic %")
    plt.xlabel("Year")
    plt.xticks(x_axis, years, fontsize=9)
    # plt.yticks(np.arange(0, 101, 10))
    plt.ylim([0, 1.0])
    # Legends
    patches = []
    topics = []
    topicname = ["system","office","sales","accounting","english","representative","software","consulting","banking","management"]
    for index, (topic, color) in enumerate(colors_dict.items()):
        patches.append(mpatches.Patch(color=color, label='Topic %s' % str(topic)))
        topics.append(topicname[topic])
    plt.legend(tuple(patches), tuple(topics), loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=5, fontsize=9,
             handlelength=0.7)
    plt.savefig("../figure/trend.png",
              bbox_inches='tight')
    plt.clf()

def trend2():


    try:
        with open("../dump/hierarchy.pickle","r") as f:
            x = pickle.load(f)
    except:
        try:
            with open("../dump/lda.pickle","r") as f:
                x = pickle.load(f)
        except:
            x = JobResume()     # Load data
            x.prepare()
            x.num_lda_topics = 100
            x.lda(content="job")
            with open("../dump/lda.pickle","w") as f:
                pickle.dump(x,f)

        num_jobs = 10
        x.hierarchy(n=num_jobs)
        with open("../dump/hierarchy.pickle","w") as f:
            pickle.dump(x,f)

    years = sorted(list(set(x.jobs["Year"])))
    result = []
    for year in years:
        set_trace()
        indices = np.where(x.jobs["Year"]==year)[0]
        total = len(indices)
        tmp = Counter(x.classes[indices])
        row = {"Year": year}
        for i in xrange(num_jobs):
            row[i]=tmp[i]
            row[i+num_jobs] = float(tmp[i])/total
        result.append(row)
    df = pd.DataFrame(result)
    df.to_csv("../figure/trend_hierarchy.csv")

def manual_triplet():
    x = JobResume()  # Load data
    x.prepare()  # Preprocessing
    x.tfidf()
    size = 100
    target_job = x.gen_inclass_triplet(category="Software Engineer",is_target_job=True,num=size)
    target_resume = x.gen_inclass_triplet(category="Software Engineer",is_target_job=False,num=size)
    columns = ["Triplet", "Job", "Resume A", "Resume B"]
    dict = {column:[] for column in columns}
    for triplet in target_job:
        dict["Triplet"].append(triplet)
        dict["Job"].append(x.job_post[triplet[0]])
        dict["Resume A"].append(x.resume_info[triplet[1]])
        dict["Resume B"].append(x.resume_info[triplet[2]])
    df = pd.DataFrame(data=dict,columns=columns)
    df.to_csv("../dump/target_job_triplet_SE.csv")

    columns = ["Triplet", "Resume", "Job A", "Job B"]
    dict = {column: [] for column in columns}
    for triplet in target_resume:
        dict["Triplet"].append(triplet)
        dict["Resume"].append(x.resume_info[triplet[0]])
        dict["Job A"].append(x.job_post[triplet[1]])
        dict["Job B"].append(x.job_post[triplet[2]])
    df = pd.DataFrame(data=dict, columns=columns)
    df.to_csv("../dump/target_resume_triplet_SE.csv")

def separate():
    x=JobResume()
    categories = ["Machinist","Machine Operator","Technician","Technologist","Engineer","Software Engineer"]

    for cat in categories:
        jobs = x.jobs.iloc[[i for i, cats in enumerate(x.jobs["categories"]) if cat in cats]]
        resumes = x.resumes.iloc[[i for i, cats in enumerate(x.resumes["categories"]) if cat in cats]]
        jobs.to_csv("../data/separate/job_"+cat+".csv", index=True)
        resumes.to_csv("../data/separate/resume_" + cat + ".csv", index=True)



def lda_distribution():
    x = JobResume()     # Load data
    x.prepare()         # Preprocessing
    x.lda(seed=0,num_topics=10,norm=None)
    x.feature_distribution()


if __name__ == "__main__":
    eval(cmd())

