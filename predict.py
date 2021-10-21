#!/usr/bin/env python
# coding: utf-8


import sys,os
import pandas as pd
from autogluon.tabular import TabularPredictor
import pickle
import numpy as np
import networkx as nx
from graphviz import Graph as DOTGraph
from collections import Counter


predictor = TabularPredictor.load("./FINAL_automl_600_auc_diffmsa01/")



alphabet='CHDENSTKGQYLAVRIMFWP-'
fqseq='CHDENSTKGQYLAVRIMFWP-'
colsLabel=[]
for idx,aa in enumerate(fqseq):
    for idx2,aa2 in enumerate(fqseq):
        colsLabel.append(''.join((aa,aa2)))
colsLabel.append('metal')

states = len(alphabet)
a2n = {}
for a,n in zip(alphabet,range(states)):
    a2n[a] = n

def aa2num(aa):
    if aa in a2n: return a2n[aa]
    else: return a2n['-']
def parse_aln2seqmtx(filename,limit=-1):
    sequence = []
    f = open(filename, "r")
    for line in f.readlines():
        if line[0]=='>':
            continue
        line = line.strip()
        sequence.append(line)
    f.close()
    return np.array(sequence)

def frequency_matrix(seqs_mtx,i,j):
    fq_mtx=np.zeros((21,21))
    i=int(i)-1
    j=int(j)-1
    for r in range(seqs_mtx.shape[0]):   
        iaa=seqs_mtx[r][i]
        jaa=seqs_mtx[r][j]
        fq_mtx[aa2num(iaa),aa2num(jaa)]+=1
    return fq_mtx

def get_frequency_mtx(fas_path,i,j):
    m=parse_aln2seqmtx(fas_path)
    fq_mtx=frequency_matrix(m,i,j)
    fq_mtx=fq_mtx/np.sum(fq_mtx)
    return fq_mtx


def main(gene,msa_file,contact_file):
    if os.path.getsize(msa_file) ==0:
        print( gene,'blank msafile')
        return
    CHED_pair=['HH','HC','CH','CC','HD','DD','DH','CD','DC','EE','EC','CE','DE','ED','HE','EH']
    coevolution=[]
    state=True
    contact_handle=open(contact_file,'r')
    contact_handle.readline() 
    for line in contact_handle.readlines():
        line=line.strip()
        if len(line)==0:
            state=False
            break
        aa_i=line.split()[-2]
        aa_j=line.split()[-1]
        i_res=aa_i[0]
        j_res=aa_j[0]        
        pair=i_res+j_res
        #print(pair)
        p=float(line.split()[2])
        if (pair in CHED_pair) and (p > 0.1):
            coevolution.append((aa_i,aa_j))
    if state==False:
        print( gene,'blank confile')
        return
    print(coevolution)
    contact_handle.close()

    pairProbaDct={}
    output_file='%s.cnn'%gene 
    output_handle=open(output_file,'w')
    for pair in coevolution:
        i=pair[0].split('_')[1]
        j=pair[1].split('_')[1]
        data=get_frequency_mtx(msa_file,i,j)
        x_prediction = data.reshape(1,441)  
        test_data=pd.DataFrame(x_prediction)
        test_data.columns=colsLabel[:-1]
        weighted_prediction = predictor.predict_proba(test_data,as_pandas=False)        
        pred = weighted_prediction[:,1]
        iAA=i+pair[0][0]
        jAA=j+pair[1][0]
        if pred > 0.5: 
            pairProbaDct[(iAA,jAA)]=pred
        output_handle.write(iAA+"\t"+jAA+"\t"+str(pred)+"\n")        
    output_handle.close()    
    with open('%s.pickle'%gene,'wb') as f:
        pickle.dump(pairProbaDct,f)



def color_def(elem):
    if elem=="C":
        color="#FFED97"
    elif elem=="H":
        color="#66B3FF"
    elif elem=="E":
        color="#FFA6FF"
    elif elem =="D":
        color="#D3A4FF"
    return color

def find_ring(subG):
    g=nx.Graph(subG)
    while ([d<2 for (n,d) in list(g.degree())].count(True)>0):
        for (n,d) in list(g.degree()):
            if d<2:
                g.remove_node(n)
    return g

def resi_comparison(G1,G2):
    return G1['residue']==G2['residue']
def gap_comparison(G1,G2):
    return G1['gap_label']==G2['gap_label']



def motif_scan(G1,stdout):
    f=open('./ModelTraining/motif_bank.pkl','rb')
    motif_bank=pickle.load(f)
    f.close()
    result=[]
    for subG_metal in list(motif_bank.keys()):
        graph_list=motif_bank[subG_metal]
        for count,G2 in enumerate(graph_list):
            GM = nx.isomorphism.GraphMatcher(G1,G2[1],node_match=resi_comparison, edge_match=gap_comparison)
            if GM.is_isomorphic():  
                result.append(subG_metal)
                print("LOG",subG_metal,count,file=stdout)
                print("LOG",GM.mapping,file=stdout)
    result=dict(Counter(result))
    for k in list(result.keys()):
        ratio="%.2f"%(result[k]/len(motif_bank[k]))
        print("MOTIF",k,result[k],len(motif_bank[k]),ratio,file=stdout)


def option(gene,N=1):
    GT_export=open("%s.dat"%gene,'w')
    with open('%s.pickle'%gene,'rb') as f:
        pairProbaDct=pickle.load(f)
    edgelist=list(pairProbaDct.keys())
    print("G",edgelist,file=GT_export)
    G=nx.from_edgelist(edgelist)
    buffer=[]
    for c in nx.connected_components(G):
        subG=nx.Graph(G.subgraph(c))
        subG=find_ring(subG)
        pair_list=list(subG.edges)
        pair_list=[(i,j) if int(i[:-1])<int(j[:-1]) else (j,i) for (i,j) in pair_list]
        buffer.append((subG.number_of_nodes(),pair_list))
        for pair in pair_list:
            attrs={pair[0]:{'residue':pair[0][-1],'id':pair[0]},pair[1]:{'residue':pair[1][-1],'id':pair[1]}}
            nx.set_node_attributes(subG,attrs)
            gap=abs(int(pair[0][:-1])-int(pair[1][:-1]))
            attrs={(pair[0],pair[1]):{'gap':gap,'gap_label': (gap<3)}}
            nx.set_edge_attributes(subG,attrs) 
        print("subG",pair_list,file=GT_export)
        motif_scan(subG,GT_export)        
    afterFilter=[]
    if sum([x[0] for x in buffer])>=N:
        for n,pair_list in buffer:
            for pair in pair_list:
                afterFilter.append("%s-%s"%(pair[0],pair[1]))
    else:
        afterPair=[p for x in buffer for p in x[1] ]
        also_ran=set(edgelist)-set(afterPair)
        pickPair=sorted(also_ran,key=lambda x:float(pairProbaDct[x]),reverse=True)[:min(N,len(also_ran))]
        for pair in pickPair:
            afterFilter.append("%s-%s"%(pair[0],pair[1]))
        for pair in afterPair:
            afterFilter.append("%s-%s"%(pair[0],pair[1]))
    for p in afterFilter:
        print("FINAL",p,file=GT_export)
    GT_export.close()
            
    dot = DOTGraph(comment=gene,format='eps')
    for pair in afterFilter: 
        print(pair)
        iaa=pair.split('-')[0]
        jaa=pair.split('-')[1]
        dot.node(iaa,style="radial",fillcolor=color_def(iaa[-1]))
        dot.node(jaa,style="radial",fillcolor=color_def(jaa[-1]))
        gap=abs(int(iaa[:-1])-int(jaa[:-1]))
        if gap < 3:
            penwidth= '2' 
        else:
            penwidth= '1'
        dot.edge(iaa,jaa,penwidth=penwidth)        
    dot.render('%s.gv'%gene,view=False)   
                

#main('P0A6G5','./P0A6G5.a3m','./P0A6G5.csv')
#option('P0A6G5')


protein_name=sys.argv[1]
MSA_file=sys.argv[2]
coevolution_file=sys.argv[3]

main(protein_name,MSA_file,coevolution_file)
option(protein_name)


