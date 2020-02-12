"""
Finding the worst case error percentage for Polynomial Codes.
Having a random matrices A and B
We have n workers, s stragglers.
Storage fraction gamma = 1/k ; where k = n - s; and set Delta = k.
Matrix A is divided into k block columns.
We choose n nodes uniformly spaced in [-1,1], instead of the integers.
One can find MSE at different SNR

This code uses the approach of the following paper-

Qian Yu, Mohammad Maddah-Ali, and Salman Avestimehr. Polynomial codes: an 
optimal design for highdimensional coded matrix multiplication. In Proc. of 
Advances in Neural Information Processing Systems (NIPS), pages 4403–4413, 2017
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
    n = 18 ;                            # Number of worker nodes
    kA = 5;
    kB = 3;
    k = kA*kB;
    s = n - k ;
    r = 9000 ;
    t = 3000 ;
    w = 6000;
    mu = 0;
    sigma = 10;
    A = np.random.normal(mu, sigma, [t,r]);
    B = np.random.normal(mu, sigma, [t,w]);
    E = np.matmul(np.transpose(A),B);
    worst_case = 1;
    node_points = -1+2*(np.array(list(range(n)))+1)/n;
    workers = np.array(list(range(n)));
    Choice_of_workers = list(it.combinations(workers,k));
    size_total = np.shape(Choice_of_workers);            
    total_no_choices = size_total[0];
    cond_no = np.zeros(total_no_choices,dtype = float);    
    SNR = 100;                          # One can change the SNR

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

