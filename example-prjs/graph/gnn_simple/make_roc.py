import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import mplhep as hep
plt.style.use(hep.style.CMS)

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score

import os
import glob
import argparse
import sys
from shutil import copyfile

from tempfile import mkstemp
from shutil import move, copymode
from os import fdopen, remove

import re

def replace(file_path, pattern, subst):
    #Create temp file
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                new_file.write(re.sub(pattern,subst,line))
                               
    #Copy the file permissions from the old file to the new file
    copymode(file_path, abs_path)
    #Remove original file
    remove(file_path)
    #Move new file
    move(abs_path, file_path)

runCsim = True
n_edges_max = 148

aucs = []
bits = [10, 11, 12, 13, 14, 15, 16, 18, 20]
for bit in bits:
    res = []
    prd = []
    tgt = []
    for f in ['parameters.h']+glob.glob('weights/*.h'):
        replace(f,r'ap_fixed<[0-9]{1,2},6>','ap_fixed<%i,6>'%bit)

    os.makedirs('tb_data_%i'%bit, exist_ok=True)
    for i in range(6*16): #90623
        if runCsim and not os.path.isfile('tb_data_%i/tb_output_edge_predictions_%05d.dat'%(bit,i)):
            copyfile('tb_data/tb_input_edge_features_%05d.dat'%i, './tb_input_edge_features.dat')
            copyfile('tb_data/tb_input_node_features_%05d.dat'%i, './tb_input_node_features.dat')
            copyfile('tb_data/tb_output_edge_predictions_%05d.dat'%i, './tb_output_edge_predictions.dat')
            copyfile('tb_data/tb_output_edge_targets_%05d.dat'%i, './tb_output_edge_targets.dat')
            copyfile('tb_data/tb_receivers_%05d.dat'%i, './tb_receivers.dat')
            copyfile('tb_data/tb_senders_%05d.dat'%i, './tb_senders.dat')
            os.system('source /xilinx/Vivado/2019.2/settings64.sh; vivado_hls -f build_prj.tcl')
            copyfile('myproject_prj/solution_roc/csim/build/tb_output_edge_labels.dat', 'tb_data_%i/tb_output_edge_labels_%05d.dat'%(bit,i))
            
        with open('tb_data/tb_output_edge_predictions_%05d.dat'%(i), 'r') as prd_file:
            for line in prd_file.readlines():
                prd_list = [float(x) for j, x in enumerate(line.split(' ')) if (j < n_edges_max) if x.strip()]
            n_edges = len(prd_list)
            prd.extend(prd_list)

        with open('tb_data/tb_output_edge_targets_%05d.dat'%(i), 'r') as tgt_file:
            for line in tgt_file.readlines():
                tgt_list = [int(float(x)) for j, x in enumerate(line.split(' ')) if (j < n_edges_max) if x.strip()]
            tgt.extend(tgt_list)

            with open('tb_data_%i/tb_output_edge_labels_%05d.dat'%(bit,i), 'r') as res_file:
                for line in res_file.readlines():
                    res_list = [float(x) for j, x in enumerate(line.split(' ')) if (j < n_edges) if x.strip()]
                res.extend(res_list)

                
    result = np.array(res)
    prdctn = np.array(prd)
    target = np.array(tgt)

    print('target\n', target[target==0])
    print('prdctn\n', prdctn[target==0])
    print('result\n', result[target==0])
    print("result shape: ", result.shape)
    print("prediction shape: ", prdctn.shape)
    print("target shape: ", target.shape)


    plt.figure()
    plt.hist(result, bins=np.linspace(0,1,51),weights=target,label='true',histtype='step',density=True)
    plt.hist(result, bins=np.linspace(0,1,51),weights=1-target,label='false',histtype='step',density=True)
    plt.semilogy()
    plt.legend(title='csim',loc='upper left')
    plt.xlabel('edge weight')
    plt.savefig('dist_%i.pdf'%bit)
    plt.figure()
    plt.hist(prdctn, bins=np.linspace(0,1,51),weights=target,label='true',histtype='step',density=True)
    plt.hist(prdctn, bins=np.linspace(0,1,51),weights=1-target,label='false',histtype='step',density=True)
    plt.semilogy()
    plt.legend(title='python',loc='upper left')
    plt.xlabel('edge weight')
    plt.savefig('dist_python_%i.pdf'%bit)

    fpr, tpr, thresholds = roc_curve(target, result)
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    print("CSIM AUC = ", roc_auc)
    plt.figure()
    plt.plot(fpr, tpr, label='CSIM %i bit (AUC = %0.3f)' % (bit,roc_auc))
    fpr, tpr, thresholds = roc_curve(target, prdctn)
    exp_auc = auc(fpr, tpr)
    print("PYTHON AUC = ", exp_auc)
    plt.plot(fpr, tpr, label='Python (AUC = %0.3f)' % exp_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(loc="lower right")
    plt.savefig('python_roc_%i.pdf'%bit)

    plt.figure()
    plt.scatter(prdctn,result,marker='o',alpha=0.5)
    plt.xlabel('python')
    plt.ylabel('csim')
    plt.savefig('scatter_%i.pdf'%bit)

plt.figure(figsize=(6,5))
plt.plot(bits, aucs, ls='-', lw=2, marker='o', label='hls4ml')
plt.plot([min(bits), max(bits)], [exp_auc,exp_auc], label='Expected (%.3f)'%exp_auc, ls='--', color='gray',lw=2)
plt.xlabel('Total bit precision')
plt.xlim(9,23)
plt.ylim(0.65,1.0)
plt.ylabel('AUC')
plt.legend(title='   112 nodes, 148 edges',loc='lower right')
plt.tight_layout()
plt.savefig('AUC_vs_BP.pdf')

