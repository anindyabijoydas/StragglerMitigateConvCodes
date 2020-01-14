"""
Having random matrices A and B of size (t,r) and (t,w), respectively.
Finding the worst case error percentage for our proposed scheme to get A' times B
We have n workers, s = n - kA*kB stragglers.
Storage fraction gammaA = 1/kA and gammaB = 1/kB.
Matrix A is divided into DeltaA = kA block columns.
Matrix B is divided into DeltaB = kB block columns.
One can vary SNR to check the scenario at different noise levels.
Set worst_case = 1 to find the error in the worst_case scenario.
Worst_condition_number is the maximum condition number in this scheme.
One can increase no_trials, which can help to find a better condition number.

This code uses the approach of the following paper-
%   Subramaniam, Adarsh M., Anoosheh Heidarzadeh, and Krishna R. Narayanan. 
%   "Random Khatri-Rao-Product Codes for Numerically-Stable Distributed Matrix Multiplication." 
%   In 57th Annual Allerton Conference on Communication, Control, and Computing 
%   (Allerton), pp. 253-259. IEEE, 2019.
"""

def mat_mat_best_rand(n,kA,kB,no_trials):
    condition_no = np.zeros(no_trials,dtype=float);
    Rr_A = {};
    Rr_B = {};
    RintA = np.zeros((k,kA),dtype = float);
    for i in range (0,kB):
        RintA[i*kA:(i+1)*kA,:] = np.identity(kA,dtype = float);
    RintB = np.zeros((k,kB),dtype = float);
    for i in range (0,kB):
        RintB[i*kA:(i+1)*kA,i] = 1;
    s = n - kA*kB;       

    for mm in range(0,no_trials):
        Rr_A[mm] = np.random.normal(0, 1, [s,kA]);
        R_A = np.concatenate((RintA,Rr_A[mm]), axis = 0)
        Rr_B[mm] = np.random.normal(0, 1, [s,kB]);
        R_B = np.concatenate((RintB,Rr_B[mm]), axis = 0)

        R_AB = np.zeros((n,k),dtype = float);
        for i in range(0,n):
            R_AB[i,:] = np.kron(R_A[i,:],R_B[i,:]);

        workers = np.array(list(range(n)));
        Choice_of_workers = list(it.combinations(workers,k));
        size_total = np.shape(Choice_of_workers);            
        total_no_choices = size_total[0];
        cond_no = np.zeros(total_no_choices,dtype = float);    

        for i in range (0, total_no_choices):
            dd = list(Choice_of_workers[i]); 
            Coding_matrix = R_AB[dd,:]
            cond_no[i] = np.linalg.cond(Coding_matrix);
        condition_no[mm] = np.max(cond_no);
    pos =   np.argmin(condition_no); 
    R_A = np.concatenate((RintA,Rr_A[pos]), axis = 0)
    R_B = np.concatenate((RintB,Rr_B[pos]), axis = 0)
    best_cond_min = condition_no[pos];
    return R_A,R_B,best_cond_min

import numpy as np
import itertools as it
import time
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

n = 11 ;
kA = 3;
kB = 3
k = kA*kB;
s = n - k ;

r = 3000 ;                  ## r needs to be a multiple of kA
t = 2000 ;
w = 3000;                   ## w needs to be a multiple of kB
mu = 0;
sigma = 2;
snr = 100;                  ## noise level
A = np.random.normal(mu, sigma, [t,r]);
B = np.random.normal(mu, sigma, [t,w]);
E = np.matmul(np.transpose(A),B);
no_trials = 20;             ## number of trials

(R_A,R_B,best_cond_min) = mat_mat_best_rand(n,kA,kB,no_trials);
R_AB = np.zeros((n,k),dtype = float);
for i in range(0,n):
    R_AB[i,:] = np.kron(R_A[i,:],R_B[i,:]);

workers = np.array(list(range(n)));
Choice_of_workers = list(it.combinations(workers,k));
size_total = np.shape(Choice_of_workers);            
total_no_choices = size_total[0];
cond_no = np.zeros(total_no_choices,dtype = float);    

for i in range (0, total_no_choices):
    dd = list(Choice_of_workers[i]); 
    Coding_matrix = R_AB[dd,:]
    cond_no[i] = np.linalg.cond(Coding_matrix);
  
worst_condition_number = np.max(cond_no);
pos =   np.argmax(cond_no);
worst_choice_of_workers = list(Choice_of_workers[pos]);
print('Worst condition Number is %s' % worst_condition_number)
print('Worst Choice of workers set includes workers %s ' % worst_choice_of_workers)

    
Coding_A = R_A;    
c = int(r/kA);
W1a = {};
for i in range (0,kA):
    W1a[i] = A[:,i*c:(i+1)*c];

W2a = {} ;
for i in range (0,n):
    W2a[i] = sum(Coding_A[i,j]*W1a[j] for j in range(0,kA));

Coding_B = R_B;    
d = int(w/kB);
W1b = {};
for i in range (0,kB):
    W1b[i] = B[:,i*d:(i+1)*d];

W2b = {} ;
for i in range (0,n):
    W2b[i] = sum(Coding_B[i,j]*W1b[j] for j in range(0,kB)) ;

work_product = {};
for i in range (0,n):
    work_product[i] = np.matmul(np.transpose(W2a[i]),W2b[i]);
    sig = work_product[i];
    sig_avg_db = 10*np.log10(np.mean(sig ** 2));
    noise_avg_db = sig_avg_db-snr;
    var = 10 ** (noise_avg_db/10);
    work_product[i] = sig + np.random.normal(0,np.sqrt(var),np.shape(sig));
  
worker_product = {};
for i in range (0,k):
    worker_product[i] = work_product[ worst_choice_of_workers[i]]

active_workers = worst_choice_of_workers;

ss = np.array(active_workers)
amw = ss[ss<k];
apw = ss[ss>=k];
allm = [i for i in range (0,k)]
arw = [i for i in allm if i not in amw]

fin_res = np.zeros((r,w),dtype=float)   
aa = np.zeros(np.size(amw),dtype=int)
bb = np.zeros(np.size(amw),dtype=int)

for i in range(0,np.size(amw)):
    aa[i] = np.remainder(amw[i],kA);
    bb[i] = int(np.floor(amw[i]/kA));
    fin_res[aa[i]*c:(aa[i]+1)*c,bb[i]*d:(bb[i]+1)*d] = worker_product[i]

for j in range(0,np.size(apw)):
    for i in range(0,np.size(amw)):
        RR = R_AB[apw[j],aa[i]*kB+bb[i]]*worker_product[i]
        worker_product[np.size(amw)+j] =  worker_product[np.size(amw)+j] - RR

start_time = time.time();
Coding_matrix_s = R_AB[apw,:];
Coding_matrix = np.delete(Coding_matrix_s,  aa*kB+bb, 1)
decoding_mat = np.linalg.inv(Coding_matrix)
CC = np.hstack(np.transpose(worker_product[np.size(amw)+i].ravel()) for i in range (0,np.size(apw)));
BB = np.transpose(np.vstack(CC[i*c*d:(i+1)*c*d] for i in range (0,np.size(apw))));  
decoded_blocks = np.matmul(BB,np.transpose(decoding_mat));
end_time = time.time();
print('Decoding time is %s seconds' %(end_time - start_time))

ind = 0;
(g,h) = np.shape(worker_product[0])
aa = np.zeros(np.size(arw),dtype=int)
bb = np.zeros(np.size(arw),dtype=int)
ab = np.zeros(np.size(arw),dtype=int)
for i in range(0,np.size(arw)):  
    RR = R_AB[arw[i],:];
    ab[i] = np.where(RR==1)[0]

ab = np.sort(ab)
for i in range(0,np.size(arw)):    
    reshaped_block = np.reshape(decoded_blocks[:,ind],(g,h));
    aa[i] = int(np.floor(ab[i]/kB));
    bb[i] = np.remainder(ab[i],kB);
    fin_res[aa[i]*c:(aa[i]+1)*c,bb[i]*d:(bb[i]+1)*d] = reshaped_block
    ind = ind+1;

err_ls = 100*np.square(np.linalg.norm(fin_res - E,'fro')/np.linalg.norm(E,'fro'));
print('Error Percentage is %s ' %err_ls)   