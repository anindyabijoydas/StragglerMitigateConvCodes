"""
Finding the worst case error percentage for the Ortho-Poly Codes
Having two random matrices A and B
We have n workers and s = n - kA * kB stragglers.
Storage fraction gammaA = 1/kA and gammaB = 1/kB.
Matrices A and B are divided into kA and kB block columns.

This code uses the approach of the following paper-

Fahim, Mohammad, and Viveck R. Cadambe. "Numerically stable polynomially coded 
computing." In 2019 IEEE International Symposium on Information Theory (ISIT), 
pp. 3017-3021. IEEE, 2019.
"""

from __future__ import division
import numpy as np
import itertools as it
import time
from mpi4py import MPI
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    n = 18 ;                                   # Number of workers
    kA = 5;
    kB = 3;
    k = kA*kB;
    s = n - k ;                                # Number of stragglers
    r = 9000 ;
    t = 3000 ;
    w = 6000;
    mu = 0;
    sigma = 10;
    SNR = 100;
    A = np.random.normal(mu, sigma, [t,r]);
    B = np.random.normal(mu, sigma, [t,w]);
    E = np.matmul(np.transpose(A),B);

    rho = np.zeros((n,1),dtype=float)
    TA = np.zeros((n,kA),dtype=float) 
    TB = np.zeros((n,kB),dtype=float)

    for i in range(0,n):
        rho[i] = np.cos((2*i+1)*np.pi/(2*n))
    
    for rr in range(0,kA):
        dd = np.cos(rr*np.arccos(rho))
        TA[:,rr] = dd[:,0]

    for rr in range(0,kB):
        dd = np.cos(rr*kA*np.arccos(rho))
        TB[:,rr] = dd[:,0]

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
    worst_choice_of_workers = list(Choice_of_workers[pos]);
    print('Worst condition Number is %s' % worst_condition_number)
    print('Worst Choice of workers set includes workers %s ' % worst_choice_of_workers)
         
    Coding_A = TA;    
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

    Coding_B = TB;    
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
    sending_time = np.zeros(n,dtype = float); 
    for i in range (0,n):
        Ai = W2a[i];
        Bi = W2b[i];
        start = time.time();
        comm.send(Ai, dest=i+1)
        comm.send(Bi, dest=i+1)
        end = time.time();
        sending_time[i] = end - start
        comm.send(SNR, dest=i+1)
    
    
    returning_time = np.zeros(n,dtype = float); 
    computation_time = np.zeros(n,dtype = float); 

    for i in range (0,n):
        computation_time[i] = comm.recv(source=i+1);
        work_product[i] = comm.recv(source=i+1);
        returning_time[i] = comm.recv(source=i+1);

    print('\n')
    for i in range (0,n):
        print("Sending time from the master to processor %s is %s" %(i,sending_time[i]))
    print('\n')
    for i in range (0,n):
        print("Computation time for processor %s is %s" %(i,computation_time[i]))
    print('\n')    
    for i in range (0,n):
        print("Returning time from processor %s to the master is %s" %(i,returning_time[i]))
    print('\n')
    worker_product = {};
    for i in range (0,k):
        worker_product[i] = work_product[ worst_choice_of_workers[i]]
    
    start_time = time.time();
    Coding_matrix =  T[worst_choice_of_workers,:]
    decoding_mat = np.linalg.inv(Coding_matrix)
    (g,h) = np.shape(worker_product[0])
 
    CC = np.hstack(np.transpose(worker_product[i].ravel()) for i in range (0,k));
    BB = np.transpose(np.vstack(CC[i*c*d:(i+1)*c*d] for i in range (0,k))) 

    decoded_blocks = np.matmul(BB,np.transpose(decoding_mat));
    end_time = time.time();
    print('Decoding time is %s seconds' %(end_time - start_time))

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
            final_res = np.concatenate((final_res,reshaped_block),axis=0)

        
    err_rs = 100*np.square(np.linalg.norm(final_res - E)/np.linalg.norm(E));
    print('\n')
    print('Error Percentage is %s ' %err_rs)   
    comm.Abort()

else:
    Ai = comm.recv(source=0)
    Bi = comm.recv(source=0)
    SNR = comm.recv(source=0)
    start_time = time.time()
    Wab = np.matmul(np.transpose(Ai),Bi);
    end_time = time.time();
    comp_time = end_time - start_time;
    comm.send(comp_time, dest=0)
    sig = Wab;
    sig_avg_db = 10*np.log10(np.mean(sig ** 2));
    noise_avg_db = sig_avg_db - SNR;
    var = 10 ** (noise_avg_db / 10);
    Wab = sig + np.random.normal(0,np.sqrt(var),np.shape(Wab));
    c1 = time.time();
    comm.send(Wab, dest=0)
    c2 = time.time();
    c = c2- c1;
    comm.send(c, dest=0)

