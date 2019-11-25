"""
Finding the worst case error percentage for the Reed-Solomon scheme
Having a random matrix A and a vector x
We have n workers, s stragglers.
Storage fraction gamma = 1/k ; where k = n - s; and set Delta = k.
Matrix A is divided into Delta = k block columns.
We choose n nodes uniformly spaced in [-1,1], instead of the integers.

This code uses the approach of the following paper-
%
%   Qian Yu, Mohammad Maddah-Ali, and Salman Avestimehr. Polynomial codes: 
%   an optimal design for highdimensional coded matrix multiplication. 
%   In Proc. of Advances in Neural Information Processing Systems
%   (NIPS), pages 4403–4413, 2017
"""

import numpy as np
import itertools as it
import time

n = 15 ;
s = 2 ;
k = n - s ;

SNR = 150; 
Delta = k;
q = int(Delta/k) ;

r = 7800 ;              ## r needs to be a multiple of Delta = k 
t = 4000 ;
mu = 0;
sigma = 2;
A = np.random.normal(mu, sigma, [t,r]);
x = np.random.normal(mu, sigma, [t,1]);
b = np.matmul(np.transpose(A),x);

c = int(r/Delta);
W1 = {};
for i in range (0,Delta):
    W1[i] = A[:,i*c:(i+1)*c];

node_points = -1+2*(np.array(list(range(n)))+1)/n;          ## RS nodes are uniformly spaced in [-1,1]
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
for j in range (0,k):
    Coding_matrix[j,:] = (nodes[j])**np.array(list(range(k)));
W2 = {} ;
for i in range (0,k):
    W2[i] = 0*W1[i];
    for j in range (0,k):
        W2[i] = W2[i] + Coding_matrix[i,j]*W1[j];

W3 = np.matmul(np.transpose(W2[0]),x);
for i in range (1,k):
    W3 = np.concatenate((W3,np.matmul(np.transpose(W2[i]),x)),axis=1);
for i in range (0,k):
    sig = W3[:,i];
    sig_avg_db = 10*np.log10(np.mean(sig ** 2));
    noise_avg_db = sig_avg_db - SNR;
    var = 10 ** (noise_avg_db / 10);
    W3[:,i] = W3[:,i] + np.random.normal(0,np.sqrt(var),np.shape(W3[:,i]));

start_time = time.time();   
 
W4 = np.matmul(np.linalg.inv(Coding_matrix),np.transpose(W3));
final_result = np.reshape(W4, (r, 1));              ## Decoded Final Result

end_time = time.time();
print('Decoding time is %s seconds' %(end_time - start_time))
err_ls = 100*np.square(np.linalg.norm(final_result - b,'fro')/np.linalg.norm(b,'fro'));
print('Error Percentage is %s ' %err_ls)   