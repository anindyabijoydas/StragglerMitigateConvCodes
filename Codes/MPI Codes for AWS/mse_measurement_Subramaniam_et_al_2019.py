"""
Finding the worst case error percentage for the Random KR Codes
Having two random matrices A and B
We have n workers and s = n - kA * kB stragglers.
Storage fraction gammaA = 1/kA and gammaB = 1/kB.
Matrices A and B are divided into kA and kB block columns.
One can change the number of trials to find better coefficients.
One can change SNR to check MSE which reflects the issue of condition number.
Results will be different for different runs, because of different sets of coefficients.

This code uses the approach of the following paper-

Subramaniam, Adarsh M., Anoosheh Heidarzadeh, and Krishna R. Narayanan. 
"Random Khatri-Rao-Product Codes for Numerically-Stable Distributed Matrix Multiplication." 
In 2019 57th Annual Allerton Conference on Communication, Control, and Computing (Allerton), 
pp. 253-259. IEEE, 2019.
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
    print('\n')
    for mm in range(0,no_trials):
        print('Trial %s is running' % mm)
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

if rank == 0:
    n = 18 ;                                        # Number of worker nodes
    kA = 5;
    kB = 3;
    k = kA*kB;
    s = n - k ;                                     # Number of stragglers
    r = 900 ;
    t = 150 ;
    w = 900;
    mu = 0;
    sigma = 1;
    A = np.random.normal(mu, sigma, [t,r]);
    B = np.random.normal(mu, sigma, [t,w]);
    E = np.matmul(np.transpose(A),B);
    SNR = 50;                                       # SNR

    no_trials = 10;
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
    (uu,vv) = np.shape(W1a[0]);
    for i in range (0,n):
        W2a[i] = np.zeros((uu,vv),dtype=float); 
        for j in range (0,kA):
            W2a[i] = W2a[i] + Coding_A[i,j]*W1a[j];

    Coding_B = R_B;    
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
    active_workers = worst_choice_of_workers;
    worker_product1 = {};

    for i in range(0,np.size(active_workers)):
        worker_product1[i] = worker_product[i] 
        
    
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
        fin_res[aa[i]*c:(aa[i]+1)*c,bb[i]*d:(bb[i]+1)*d] = worker_product1[i]

    for j in range(0,np.size(apw)):
        for i in range(0,np.size(amw)):
            RR = R_AB[apw[j],aa[i]*kB+bb[i]]*worker_product1[i]
            worker_product1[np.size(amw)+j] =  worker_product1[np.size(amw)+j] - RR

    start_time1 = time.time();
    Coding_matrix_s = R_AB[apw,:];
    Coding_matrix1 = np.delete(Coding_matrix_s,  aa*kB+bb, 1)
    decoding_mat1 = np.linalg.inv(Coding_matrix1)
    CC1 = np.hstack(np.transpose(worker_product1[np.size(amw)+i].ravel()) for i in range (0,np.size(apw)));
    BB1 = np.transpose(np.vstack(CC1[i*c*d:(i+1)*c*d] for i in range (0,np.size(apw))));  
    decoded_blocks1 = np.matmul(BB1,np.transpose(decoding_mat1));
    end_time1 = time.time();
    print('Decoding time is %s seconds' %(end_time1 - start_time1))
    print('\n')
    ind = 0;
    (g,h) = np.shape(worker_product1[0])
    aa = np.zeros(np.size(arw),dtype=int)
    bb = np.zeros(np.size(arw),dtype=int)
    ab = np.zeros(np.size(arw),dtype=int)
    for i in range(0,np.size(arw)):  
        RR = R_AB[arw[i],:];
        ab[i] = np.where(RR==1)[0]

    ab = np.sort(ab)
    for i in range(0,np.size(arw)):    
        reshaped_block1 = np.reshape(decoded_blocks1[:,ind],(g,h));
        aa[i] = int(np.floor(ab[i]/kB));
        bb[i] = np.remainder(ab[i],kB);
        fin_res[aa[i]*c:(aa[i]+1)*c,bb[i]*d:(bb[i]+1)*d] = reshaped_block1
        ind = ind+1;

    err_rg = 100*np.square(np.linalg.norm(fin_res - E)/np.linalg.norm(E));
    print('Error Percentage is %s ' %err_rg)   
    
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

