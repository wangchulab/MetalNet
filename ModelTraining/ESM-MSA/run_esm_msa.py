import matplotlib.pyplot as plt
import esm
import torch
import os
from Bio import SeqIO
import itertools
from typing import List, Tuple
import string
import numpy as np

torch.set_grad_enabled(False)

# This is an efficient way to delete lowercase characters and insertion characters from a string
deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)

def read_sequence(filename: str) -> Tuple[str, str]:
    """ Reads the first (reference) sequences from a fasta or MSA file."""
    record = next(SeqIO.parse(filename, "fasta"))
    return record.description, str(record.seq)

def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    return sequence.translate(translation)

def read_msa(filename: str, nseq: int) -> List[Tuple[str, str]]:
    """ Reads the first nseq sequences from an MSA file, automatically removes insertions."""
    return [(record.description, remove_insertions(str(record.seq)))
            for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)]

def get_nseq( fn ):
    lines = open(fn, 'r').readlines()
    return len(lines), len(lines[1])-1

msa_transformer, msa_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
msa_transformer = msa_transformer.eval().cuda()
msa_batch_converter = msa_alphabet.get_batch_converter()

lines = open("all.list", 'r').readlines()
for l in lines:
    dn = "data/" + l[1:3] + "/"
    fn = dn + l.strip() + ".64.a3m"
    nseq, Lseq = get_nseq( fn )
    print(fn, nseq, Lseq)
    if Lseq > 1023: continue

    flag = True
    while flag:
        try:
            msa_data = []
            msa_data.append( read_msa(fn, nseq) )
            msa_batch_labels, msa_batch_strs, msa_batch_tokens = msa_batch_converter(msa_data)
            msa_batch_tokens = msa_batch_tokens.cuda()
            print(msa_batch_tokens.size())  # Should be a 3D tensor with dtype torch.int64.
            msa_contacts = msa_transformer.predict_contacts(msa_batch_tokens).cpu()
            flag = False
        except:
            nseq = int(nseq/2)


    #save output
    np.savetxt(dn + l.strip() + ".64.mtx", msa_contacts[0,:,:].numpy())
    with open(dn + l.strip() + ".64.csv", 'w') as fp:
        fp.write("i j P_ij i_aa j_aa\n")
        for contact, msa in zip(msa_contacts, msa_batch_strs):
            seq = msa[0]
            N = len(seq)
            print(seq, N)
            for i in range(N-2):
                AA_i = seq[i]
                for j in range(i+2, N):
                    AA_j = seq[j]
                    P_ij = contact[i, j]
                    if P_ij >= 0.1:
                        ostr = "%d %d %f %s %s\n" % (i, j, P_ij, AA_i+"_"+str(i+1), AA_j+"_"+str(j+1))
                        fp.write( ostr )

