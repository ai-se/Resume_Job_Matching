from __future__ import print_function
from __future__ import absolute_import, division
from random import randint,random, uniform
from time import time
import numpy as np
from pdb import set_trace
from demos import cmd
from jobfree import JobResume

class Model(object):
    def any(self):
        while True:
            for i in range(0,self.decnum):
                self.dec[i]=uniform(self.bottom[i],self.top[i])
            if self.check(): break
        return self

    def __init__(self):
        self.bottom=[0]
        self.top=[0]
        self.decnum=0
        self.dec=[]
        self.lastdec=[]
        self.obj=[]
        self.any()

    def eval(self):                                                                  
        return sum(self.getobj())

    def copy(self,other):
        self.dec=other.dec[:]
        self.lastdec=other.lastdec[:]
        self.obj=other.obj[:]
        self.bottom=other.bottom[:]
        self.top=other.top[:]
        self.decnum=other.decnum

    def setdec(self,dec):
        self.dec = dec

    def getobj(self):
        return []

    def getdec(self):
        return self.dec

    def check(self):
        for i in range(0,self.decnum):
            if self.dec[i]<self.bottom[i] or self.dec[i]>self.top[i]:
                return False
        return True

class Optimizee(Model):
    def __init__(self):
        self.bottom=[10,0,0]
        self.top=[200,1,1]
        self.decnum=3
        self.dec=[0]*self.decnum
        self.lastdec=[]
        self.obj=[]
        self.any()
        self.model = JobResume()     # Load data
        self.model.prepare()         # Preprocessing

        self.margin = 0.0
        self.num_triplets = 100000

    def getobj(self):
        if self.obj==[]:

            result_job = 0
            result_resume = 0

            score = 1 / float(self.num_triplets)

            seed = 0
            self.model.lda(seed=seed, num_topics = int(self.dec[0]), alpha=self.dec[1], eta=self.dec[2])
            for epoch in range(self.num_triplets):
                triplet = self.model.gen_triplet(type="job")
                diff = self.model.cos_dist(triplet[0] + self.model.num_resume, triplet[1]) - self.model.cos_dist(triplet[0] + self.model.num_resume,
                                                                                      triplet[2])
                if diff > self.margin:
                    result_job += score
                elif diff < -self.margin:
                    result_job += -score

                triplet = self.model.gen_triplet(type="resume")
                diff = self.model.cos_dist(triplet[0], triplet[1] + self.model.num_resume) - self.model.cos_dist(triplet[0],
                                                                                      triplet[2] + self.model.num_resume)
                if diff > self.margin:
                    result_resume += score
                elif diff < -self.margin:
                    result_resume += -score

            self.obj = [result_job, result_resume]
        return self.obj

def mutate(candidates,f,cr,i):
    tmp=range(len(candidates))
    tmp.remove(i)
    while True:
        abc=np.random.choice(tmp,3)
        a3=[candidates[tt] for tt in abc]
        xold=candidates[i]
        r=randint(0,xold.decnum-1)
        xnew=Optimizee()
        xnew.any()
        for j in xrange(xold.decnum):
            if random()<cr or j==r:
                xnew.dec[j]=a3[0].dec[j]+f*(a3[1].dec[j]-a3[2].dec[j])
            else:
                xnew.dec[j]=xold.dec[j]
        if xnew.check(): break
    if xnew.eval()<xold.eval():
        return xnew
    else:
        return xold


"DE, maximization"
def differential_evolution():

    import multiprocessing as mp
    # nb = mp.cpu_count()
    nb = 10
    pool = mp.Pool(nb)

    maxtries=10
    f=0.75
    cr=0.3
    candidates=[Optimizee() for i in range(nb)]
    for tries in range(maxtries):
        print(", Retries: %2d, " %tries)
        next_gen=[pool.apply(mutate, args=(candidates,f,cr,i)) for i in range(nb)]
        candidates = next_gen
    evals = [x.eval() for x in candidates]
    xbest = candidates[np.argmax(evals)]
    pool.close()
    print("Best solution: %s, " %xbest.dec,"obj: %s, " %xbest.getobj(),
          "evals: %s * %s" %(nb,maxtries))
    return xbest.dec,xbest.obj

def best_LDA():
    best_dec, best_obj = differential_evolution()
    x = Optimizee()
    x.setdec(best_dec)

    print(x.getobj())
    print("Parameters: ")
    print(best_dec)


if __name__ == "__main__":
    eval(cmd())



