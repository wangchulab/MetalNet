# MetalNet
We developed a coevolution-based machine learning method named “MetalNet” to systematically predict metal-binding sites in proteomes. Our computational method provides a unique and enabling tool for interrogating the hidden metalloproteome and studying metal biology.

Contact: <chuwang@pku.edu.cn>, <yao.cheng69@pku.edu.cn>, <wendao@pku.edu.cn>

## Requirements  
[autogluon](https://github.com/awslabs/autogluon) ==0.2.0    

numpy==1.19.5

pandas==1.1.5

[networkx](https://github.com/networkx/networkx) ==2.5.1

[graphviz](http://graphviz.org/download/)==0.8.4


## Prediction
    python  predict.py  [protein_name]  [MSA_file]  [esm_coevolution_profile]

run example:

    python  predict.py  P0A6G5  P0A6G5.a3m  P0A6G5.csv
    
and standard output of the example can be found in P0A6G5_output/.






