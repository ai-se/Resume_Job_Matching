from __future__ import print_function, division
import pandas as pd
import numpy as np
from pdb import set_trace
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing

from demos import cmd
import re

class Resume():
    def __init__(self,resumefile="../data/combinedResume_V2.csv"):
        self.resumes = pd.read_csv(resumefile)
        self.resumes["categories"] = [set(x.split(",")) for x in self.resumes["Job Title"]]
        self.resumes_info = [re.sub(r'[^a-zA-Z]', ' ', x.decode('string_escape').lower()) for x in self.resumes["Text"]]
        # self.resumes_info = [x.decode('string_escape').lower() for x in self.resumes["Text"]]

        tfer = TfidfVectorizer(lowercase=True, analyzer="word", norm=None, use_idf=False, smooth_idf=False,sublinear_tf=False,decode_error="ignore")
        self.csr_mat = tfer.fit_transform(self.resumes_info)
        self.voc = tfer.vocabulary_

    def get_gender(self,id):
        # try:
        #     the_list = self.resumes_info[id].split()
        #     col = the_list.index("gender")
        #     return the_list[col+1]
        # except:
        genders = ['male','female']
        for g in genders:
            if self.csr_mat[id, self.voc[g]]>0:
                return g
        return np.nan


    def gender(self):
        self.resumes["gender"] = [self.get_gender(id) for id in range(len(self.resumes))]


def find_gender():
    res = Resume()
    res.gender()
    df = res.resumes[res.resumes["gender"].notnull()]
    df.to_csv("../bias_data/resumes.csv", line_terminator="\r\n", index=False)


if __name__ == "__main__":
    eval(cmd())


