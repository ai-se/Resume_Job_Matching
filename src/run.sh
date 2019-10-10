#! /bin/tcsh
rm ./err/*
rm ./out/*

bsub -q long -W 8000 -n 10 -o ./out/%J.out -e ./err/%J.err /share/tjmenzie/zyu9/miniconda2/bin/python2.7 DE.py best_LDA
