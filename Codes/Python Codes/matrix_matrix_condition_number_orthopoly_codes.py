"""
Finding the worst case condition number for ortho-poly scheme
We have n workers and s = n - kA * kB stragglers.
Storage fraction gammaA = 1/kA and gammaB = 1/kB.
Matrices A and B are divided into kA and kB block columns.

This code uses the approach of the following paper-
%   M. Fahim and V. R. Cadambe, "Numerically Stable Polynomially Coded Computing," 
%   2019 IEEE International Symposium on Information Theory (ISIT), 
%   Paris, France, 2019, pp. 3017-3021.
"""
import numpy as np
import itertools as it

n = 12;                                             ## number of total workers
kA = 2;
kB = 5;
k = kA*kB;                                          ## number of total message workers
s = n - k;                                          ## number of stragglers

rho = np.zeros((n,1),dtype=float)
TA = np.zeros((n,kA),dtype=float)
TB = np.zeros((n,kB),dtype=float)

for i in range(0,n):
    rho[i] = np.cos((2*i+1)*np.pi/(2*n))
    
for r in range(0,kA):
    dd = np.cos(r*np.arccos(rho))
    TA[:,r] = dd[:,0]

for r in range(0,kB):
    dd = np.cos(r*kA*np.arccos(rho))
    TB[:,r] = dd[:,0]

TA[:,0] = TA[:,0]/np.sqrt(2);
TB[:,0] = TB[:,0]/np.sqrt(2);    

T = np.zeros((n,kA*kB),dtype=float)
for i in range(0,n):
    T[i,:] = np.kron(TA[i,:],TB[i,:])
    
workers = list(range(n));
Choice_of_workers = list(it.combinations(workers,k));
size_total = np.shape(Choice_of_workers);            
total_no_choices = size_total[0];
cond_no = np.zeros(total_no_choices,dtype = float);  

for kk in range (0, total_no_choices):
    wor = list(Choice_of_workers[kk]);
    cond_no[kk] = np.linalg.cond(T[wor,:]);    

worst_condition_number = np.max(cond_no);
pos =   np.argmax(cond_no);
active_workers = list(Choice_of_workers[pos]);     
  
print("Worst Case Condition Number is %s" %worst_condition_number)
print("Worst Case includes workers %s" %active_workers)    