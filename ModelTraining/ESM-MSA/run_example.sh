#with hhsuite3 installed
cd data/A0
hhfilter -i 2A0BA.a3m -diff 64 -cov 75 -id 90 -o 2A0BA.64.a3m
cd -

#with esm installed
python run_esm_msa.py
