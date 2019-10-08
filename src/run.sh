#! /bin/tcsh

bsub -q standard -W 2400 -n 10 -o ./out/%J.out -e ./err/%J.err /share/tjmenzie/zyu9/miniconda2/bin/python2.7 job_resume.py best_LDA
