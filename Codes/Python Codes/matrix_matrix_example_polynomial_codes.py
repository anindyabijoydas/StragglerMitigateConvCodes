"""
Finding the worst case error percentage for the Reed-Solomon scheme
Having two random matrices A and B
We have n workers and s = n - kA * kB stragglers.
Storage fraction gammaA = 1/kA and gammaB = 1/kB.
Matrices A and B are divided into kA and kB block columns.
We choose n nodes uniformly spaced in [-1,1].
Set worst_case = 1 to find the error in the worst_case scenario.

This code uses the approach of the following paper-
%   Qian Yu, Mohammad Maddah-Ali, and Salman Avestimehr. Polynomial codes: 
%   an optimal design for highdimensional coded matrix multiplication. 
%   In Proc. of Advances in Neural Information Processing Systems
%   (NIPS), pages 4403–4413, 2017
"""


import numpy as np
import itertools as it
import time

n = 14 ;
kA = 4;
kB = 3;
k = kA*kB;
s = n - k ;

r = 3000 ;                  ## r needs to be a multiple of kA
t = 4000 ;
w = 3000;                   ## w needs to be a multiple of kB
mu = 0;
sigma = 2;
snr = 100;                  ## noise level
A = np.random.normal(mu, sigma, [t,r]);
B = np.random.normal(mu, sigma, [t,w]);
E = np.matmul(np.transpose(A),B);

node_points = -1+2*(np.array(list(range(n)))+1)/n;      ## RS nodes are uniformly spaced in [-1,1]
workers = np.array(list(range(n)));
Choice_of_workers = list(it.combinations(workers,k));
size_total = np.shape(Choice_of_workers);            
total_no_choices = size_total[0];
cond_no = np.zeros(total_no_choices,dtype = float);    

for i in range (0, total_no_choices):
    dd = list(Choice_of_workers[i]); 
    nodes = node_points[dd];
    Coding_matrix = np.zeros((k,k),dtype = float);
    for j in range (0,k):
        Coding_matrix[j,:] = (nodes[j])**np.array(list(range(k)))
    cond_no[i] = np.linalg.cond(Coding_matrix);
  
worst_condition_number = np.max(cond_no);
pos =   np.argmax(cond_no);
worst_choice_of_workers = list(Choice_of_workers[pos]);
print('Worst condition Number is %s' % worst_condition_number)
print('Worst Choice of workers set includes workers %s ' % worst_choice_of_workers)

nodes = node_points[worst_choice_of_workers];
Coding_matrix = np.zeros((k,k),dtype = float);
Coding_matrix1 = np.zeros((n,k),dtype = float);


for j in range (0,k):
    Coding_matrix[j,:] = (nodes[j])**np.array(list(range(k)));

for j in range (0,n):
    Coding_matrix1[j,:] = (node_points[j])**np.array(list(range(k)));
#
    
Coding_A = Coding_matrix1[:,::kB];    
c = int(r/kA);
W1a = {};
for i in range (0,kA):
    W1a[i] = A[:,i*c:(i+1)*c];

W2a = {} ;
(uu,vv) = np.shape(W1a[0]);
for i in range (0,n):
    W2a[i] = np.zeros((uu,vv),dtype=float); 
    for j in range (0,kA):
        W2a[i] = W2a[i] + Coding_A[i,j]*W1a[j];

Coding_B = Coding_matrix1[:,0:kB];    
d = int(w/kB);
W1b = {};
for i in range (0,kB):
    W1b[i] = B[:,i*d:(i+1)*d];

W2b = {} ;
(uu,vv) = np.shape(W1b[0]);
for i in range (0,n):
    W2b[i] = np.zeros((uu,vv),dtype=float); 
    for j in range (0,kB):
        W2b[i] = W2b[i] + Coding_B[i,j]*W1b[j];

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

start_time = time.time();

decoding_mat = np.linalg.inv(Coding_matrix)
(g,h) = np.shape(worker_product[0])

xx = worker_product[0];
CC = xx.ravel();
for i in range (1,k):
    xx = worker_product[i];
    yy = np.transpose(xx.ravel());
    CC = np.concatenate((CC,yy),axis = 0);

BB = np.zeros((c*d, k),dtype=float);
for i in range(0,k):
    BB[:,i] = CC[i*c*d:(i+1)*c*d];
    
decoded_blocks = np.matmul(BB,np.transpose(decoding_mat));
ind = 0;

for i in range(0,kA):    
    for j in range(0,kB):
        if j==0:
            reshaped_block = np.reshape(decoded_blocks[:,ind],(g,h));
        else:
            reshaped_block = np.concatenate((reshaped_block,np.reshape(decoded_blocks[:,ind],(g,h))), axis = 1)
        ind = ind+1;
    if i == 0:
        final_res = reshaped_block;
    else:
        final_res = np.concatenate((final_res,reshaped_block),axis=0)   ## Decoded Final Result

end_time = time.time();
print('Decoding time is %s seconds' %(end_time - start_time))
err_ls = 100*np.square(np.linalg.norm(final_res - E,'fro')/np.linalg.norm(E,'fro'));
print('Error Percentage is %s ' %err_ls)   