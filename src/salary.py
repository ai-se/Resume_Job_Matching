from __future__ import print_function, division
import pandas as pd
import numpy as np
from pdb import set_trace

from demos import cmd

import re
def hasNumbers(inputString):
    if pd.isnull(inputString):
        return False
    return bool(re.search(r'\d', inputString))

def summarize_salary():
    from collections import Counter
    file = "../data/job.csv"
    jobs = pd.read_csv(file)
    set_trace()
    m = len(jobs["Salary"])
    has_salary = filter(hasNumbers,jobs["Salary"])

    n = len(has_salary)
    print("%d in %d has salary." %(n,m))
    print(Counter(has_salary))
    set_trace()



if __name__ == "__main__":
    eval(cmd())