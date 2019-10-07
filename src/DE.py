from __future__ import print_function
from __future__ import absolute_import, division
from random import randint,random, uniform
from time import time
import numpy as np
from pdb import set_trace
from demos import cmd
from job_resume import JobResume

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
        self.jobids = {"photographer":6445,"Office Manager":3238,"HR":18993,"ASP.NET Developer":11854,"Sales/Consultant":1525,"Administrative Assistant":14386,"Graphic Designer":14585,"Software Engineer":413, "User Interface/ Web Designer":16808, "Lawyer":19000}
        self.resumeids = {"photographer":44,"Office Manager":103,"HR":3,"ASP.NET Developer":48,"Sales/Consultant":605,"Administrative Assistant":504,"Graphic Designer":1168,"Software Engineer":881,"User Interface/ Web Designer":41, "Lawyer":392}

    def getobj(self):

        result_job = []
        result_resume = []
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        proc_num = 10

        # for seed in range(10):
        self.model.lda(seed=rank, num_topics = int(self.dec[0]), alpha=self.dec[1], eta=self.dec[2])
        r_job = 0
        r_resume = 0
        for key in self.jobids:
            jobid = self.jobids[key]
            resume_yes = self.resumeids[key]
            for r in self.resumeids:
                if r==key:
                    continue
                resume_no = self.resumeids[r]
                diff = self.model.cos_dist(jobid+self.model.num_resume,resume_yes) - self.model.cos_dist(jobid+self.model.num_resume,resume_no)
                if diff > self.margin:
                    r_job+=1
                elif diff < -self.margin:
                    r_job+=-1
        for key in self.resumeids:
            resumeid = self.resumeids[key]
            job_yes = self.jobids[key]
            for r in self.jobids:
                if r==key:
                    continue
                job_no = self.jobids[r]
                diff = self.model.cos_dist(resumeid,job_yes+self.model.num_resume) - self.model.cos_dist(resumeid,job_no+self.model.num_resume)
                if diff > self.margin:
                    r_resume+=1
                elif diff < -self.margin:
                    r_resume+=-1

        if rank==0:
            for i in range(proc_num - 1):
                (r_job,r_resume) = comm.recv(source=i + 1)
                result_job.append(r_job)
                result_resume.append(r_resume)
            set_trace()
            return [np.median(result_job), np.median(result_resume)]
        else:
            comm.send((r_job,r_resume), dest=0)



"DE, maximization"
def differential_evolution(**kwargs):

    def mutate(candidates,f,cr,xbest):
        for i in xrange(len(candidates)):
            tmp=range(len(candidates))
            tmp.remove(i)
            while True:
                abc=np.random.choice(tmp,3)
                a3=[candidates[tt] for tt in abc]
                xold=candidates[i]
                r=randint(0,xold.decnum-1)
                xnew=Optimizee(**kwargs)
                xnew.any()
                for j in xrange(xold.decnum):
                    if random()<cr or j==r:
                        xnew.dec[j]=a3[0].dec[j]+f*(a3[1].dec[j]-a3[2].dec[j])
                    else:
                        xnew.dec[j]=xold.dec[j]
                if xnew.check(): break
            if xnew.eval()>xbest.eval():
                xbest.copy(xnew)
                print("!",end="")
            elif xnew.eval()>xold.eval():
                print("+",end="")
            else:
                xnew=xold
                print(".",end="")
            yield xnew


    nb=10
    maxtries=10
    f=0.75
    cr=0.3
    xbest=Optimizee(**kwargs)
    candidates=[xbest]
    for i in range(1,nb):
        x=Optimizee(**kwargs)
        candidates.append(x)
        if x.eval()>xbest.eval():
            xbest.copy(x)
    for tries in range(maxtries):
        print(", Retries: %2d, : Best solution: %s, " %(tries,xbest.dec),end="")
        candidates=[xnew for xnew in mutate(candidates,f,cr,xbest)]
        print("")
    print("Best solution: %s, " %xbest.dec,"obj: %s, " %xbest.getobj(),
          "evals: %s * %s" %(nb,maxtries))
    return xbest.dec,xbest.obj

def best_LDA():
    best_dec, best_obj = differential_evolution()
    x = Optimizee()
    x.setdec(best_dec)
    print(x.getobj())


if __name__ == "__main__":
    eval(cmd())



