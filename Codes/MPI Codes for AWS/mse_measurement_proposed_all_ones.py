"""
Having matrices A and B of size (t,r) and (t,w), respectively.
Finding the worst case error percentage for our proposed scheme to get A' times B
We have n workers, s = n - kA*kB stragglers.
Storage fraction gammaA > 1/kA and gammaB > 1/kB.
Matrix A is divided into DeltaA block columns, where kA divides DeltaA.
Matrix B is divided into DeltaB block columns, where kB divides DeltaB.
One can vary SNR to check the scenario at different noise levels.
Set worst_case = 1 to find the error in the worst_case scenario.
Worst_condition_number is the maximum condition number in this scheme.
"""

from __future__ import division
import numpy as np
import itertools as it
import scipy as sp
import time
from scipy.sparse import csr_matrix
from scipy.linalg import eigvals
from mpi4py import MPI
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def shiftrow(A,r,t):   
    (a,b) = np.shape(A) ;
    B = np.zeros((t,b),dtype = float) ;
    B[r:a+r,:] = A ;
    return B;

if rank == 0:
    n = 18;                                                 ## Total Number of workers
    kA = 5;
    kB = 3;
    k = kA*kB;
    s = n - k;                                              ## Total Number of stragglers
    gammaA = 1/4 ;
    DeltaA = int(np.round((s-1)*(kA-1)/(gammaA-1/kA))); 
    while DeltaA % kA != 0:
        DeltaA = DeltaA+1 ;
    qA = int(DeltaA/kA) ;
    print("\nThe value of kqA is %s" %(DeltaA))
    gammaB = 2/5 ;
    DeltaB = int(np.round((s-1)*(kB-1)/(gammaB-1/kB))); 
    while DeltaB % kB != 0:
        DeltaB = DeltaB+1 ;
    qB = int(DeltaB/kB) ;
    print("The value of kqB is %s" %(DeltaB))
    r = 16000 ;
    t = 20000 ;
    w = 12000;
    mu = 0;
    sigma = 10;
    A = np.random.normal(mu, sigma, [t,r]);
    B = np.random.normal(mu, sigma, [t,w]);
    E = np.matmul(np.transpose(A),B);
    normE = np.linalg.norm(E);
    worst_case = 0;
    SNR = 100;
    for i in range (0,n):
        comm.send(SNR, dest=i+1)

    R_A = np.ones((kA,s),dtype = float);
    R_B = np.ones((kB,s),dtype = float);
        
    if worst_case !=1:
        all_workers = np.random.permutation(n);
        active_workers = all_workers[0:k];
        active_workers.sort() ;
        print("\nActive Workers are %s"%(active_workers))

    aa = int(r/DeltaA);
    Wa = {};
    for i in range (0,DeltaA):
        Wa[i] = A[:,i*aa:(i+1)*aa];
    
    bb = int(w/DeltaB);
    Wb = {};
    for i in range (0,DeltaB):
        Wb[i] = B[:,i*bb:(i+1)*bb];

    Wa1 = {} ;
    for i in range (0,k):
        for j in range (0,qA):
            Wa1[i,j] = Wa[np.floor(i/kB)*qA+j];        

    for i in range (k,n):
        lenA = qA + (kA-1)*(i-k);
        for j in range (0,lenA):
            sumA = np.zeros((t,aa),dtype = float);
            for rr in range (0,kA):
                if j >= (i-k)*rr and j <= (i-k)*rr+qA-1:
                    sumA = sumA + Wa1[kB*rr,j-(i-k)*rr]*R_A[rr,i-k];                
            Wa1[i,j] = sumA;

    Wbb = {};
    for i in range (0,kB):
        for j in range (0,qB):
            Wbb[i,j] = Wb[i*qB+j];
    Wb1 = {};
    for i in range (0,k):
        for j in range (0,qB):
            Wb1[i,j] = Wbb[np.remainder(i,kB),j];

    for i in range (k,n):
        lenB = qB + (kB-1)*(i-k);
        for j in range (0,lenB):
            sumB = np.zeros((t,bb),dtype = float);
            for rr in range (0,kB):
                if j >= (i-k)*rr and j <= (i-k)*rr+qB-1:
                    sumB = sumB + Wbb[rr,j-(i-k)*rr]*R_B[rr,i-k];                
            Wb1[i,j] = sumB;
    Wab = {};
    WW = {};
    for i in range (0,n):
        lengA = qA if i < k else qA + (kA-1)*(i-k) ;
        WsendA = {};
        for j in range (0,lengA):
            WsendA[j] = Wa1[i,j];
        comm.send(WsendA, dest=i+1)
        comm.send(lengA, dest=i+1)
        lengB = qB if i < k else qB + (kB-1)*(i-k) ;
        WsendB = {};
        for j in range (0,lengB):
            WsendB[j] = Wb1[i,j];
        comm.send(WsendB, dest=i+1)
        comm.send(lengB, dest=i+1)
        
    M1 = {} ;
    for i in range (0,kA):
        coding_matrixa = np.zeros((qA,DeltaA),dtype=float);
        coding_matrixa[:,i*qA:(i+1)*qA] = np.identity(qA,dtype=float);
        M1[i] = coding_matrixa;
    zer = 2*DeltaA; 
    for i in range (k,n):
        M1[i] = np.zeros((zer,DeltaA),dtype =float);
        for j in range (0,kA):
            coding_matrixa = shiftrow(R_A[j,i-k]*M1[j],j*(i-k),zer);
            M1[i] = M1[i] + coding_matrixa;
        inter = M1[i];
        M1[i] = inter[~np.all(inter == 0, axis=1)]   

    M_A = {} ;
    for i in range (0,n):
        if i >= k :
            M_A[i] = M1[i];
        else:
            M_A[i] = M1[np.floor(i/kB)];

    M2 = {} ;
    for i in range (0,kB):
        coding_matrixb = np.zeros((qB,DeltaB),dtype=float);
        coding_matrixb[:,i*qB:(i+1)*qB] = np.identity(qB,dtype=float);
        M2[i] = coding_matrixb;
    zer = 2*DeltaB; 
    for i in range (k,n):
        M2[i] = np.zeros((zer,DeltaB),dtype =float);
        for j in range (0,kB):
            coding_matrixb = shiftrow(R_B[j,i-k]*M2[j],j*(i-k),zer);
            M2[i] = M2[i] + coding_matrixb;
        inter = M2[i];
        M2[i] = inter[~np.all(inter == 0, axis=1)]   

    M_B = {} ;
    for i in range (0,n):
        if i >=k :
            M_B[i] = M2[i];
        else:
            M_B[i] = M2[np.remainder(i,kB)];

    P = {};
    for i in range (0,n):
        [la1,la2]=np.shape(M_A[i]);
        [lb1,lb2]=np.shape(M_B[i]);
        P[i] = np.hstack((np.vstack(np.kron(M_A[i][j,:],M_B[i][mm,:]) for mm in range(0,lb1))).T for j in range(0,la1))
   
    if worst_case ==1:
        workers = list(range(n));
        Choice_of_workers = list(it.combinations(workers,k));
        size_total = np.shape(Choice_of_workers);            
        total_no_choices = size_total[0];
        cond_no = np.zeros(total_no_choices,dtype = float);  
        for i in range (0, total_no_choices):
            kk = list(Choice_of_workers[i]);
            for j in range (0,k):
                if j==0:
                    Coding_matrix = P[kk[j]];
                else:
                    Coding_matrix = np.concatenate((Coding_matrix, P[kk[j]]), axis = 1);
            cond_no[i] = np.linalg.cond(Coding_matrix);    
        worst_condition_number = np.max(cond_no);
        pos = np.argmax(cond_no);
        active_workers = list(Choice_of_workers[pos]);       
        print("\n")           
        print("Worst Case  Condition Number is %s"%(worst_condition_number))
        print("Active Workers are %s"%(active_workers))        

    ss = np.array(active_workers)
    amw = ss[ss<k];
    apw = ss[ss>=k];
    WW = {};
    Wab = {};
    print("\n")
    for i in range (0,n):
        WW = comm.recv(source=i+1);

        ind = 0;
        la1 = qA+np.max([(i-k)*(kA-1),0]);
        for j in range (0,la1):
            lb1 = qB+np.max([(i-k)*(kB-1),0]);
            for m in range (0,lb1):
                Wab[i,ind] = WW[ind];
                ind = ind + 1;

    final_r = np.zeros((r,w),dtype=float);
    for  ii in range (0,np.size(amw)):
        jj = amw[ii];
        r_ind = np.floor(jj/kB)+1;
        c_ind = np.remainder(jj+1,kB);
        ind = 0;
        if c_ind==0:
            c_ind = kB;
        r1 = (r_ind-1)*qA
        c1 = (c_ind-1)*qB
        for i in range(0,qA):
            for j in range (0,qB):
                cur = Wab[jj,ind];
                r_arr1 = int((r1+i)*aa)
                r_arr2 = int((r1+i+1)*aa);
                c_arr1 = int((c1+j)*bb)
                c_arr2 = int((c1+j+1)*bb);
                final_r[r_arr1:r_arr2,c_arr1:c_arr2] = cur;
                ind = ind+1;

    if np.size(apw)>0:
        Coding_matrix = np.vstack(np.transpose(P[active_workers[np.size(amw)+j]]) for j in range(0,np.size(apw)))    

    [xx,yy]=np.shape(Coding_matrix)
    if xx > 0:
        for i in range (0,k):
            if i in amw:
                r_ind = np.floor(i/kB)+1;
                c_ind = np.remainder(i+1,kB); 
                if c_ind==0:
                    c_ind = kB;
                for ii in range(0,qA):
                    for jj in range(0,qB):
                        Coding_matrix[:,int(((r_ind-1)*qA+(ii))*DeltaB+(c_ind-1)*qB+jj)]=0;

    inter = np.transpose(Coding_matrix);
    unknowns = np.where(inter.any(axis=1))[0]
    Coding_matrix = np.transpose(inter[~np.all(inter == 0, axis=1)])  
    
    for i in range(k,n):
        if i not in apw:
            continue;
        lenA = qA + (kA - 1)*(i-k);
        lenB = qB + (kB - 1)*(i-k);
        ind = 0;
    
        for j in range (0,lenA):
            for jj in range (0,lenB):
                sumA = np.zeros((aa,bb),dtype = float)
                for rr in range (0,k):
                    if rr not in amw:
                        continue
                    r_ind = np.floor(rr/kB)+1;
                    c_ind = np.remainder(rr+1,kB); 
                    if c_ind==0:
                        c_ind = kB;
                    if j >= (i-k)*(r_ind-1) and j <= (i-k)*(r_ind-1)+qA-1:
                        if jj >= (i-k)*(c_ind-1) and jj <= (i-k)*(c_ind-1)+qB-1:
                            r1 = j - (i-k)*(r_ind-1);
                            c1 = jj - (i-k)*(c_ind-1);
                            sumA = sumA + Wab[rr,r1*qB+c1]*R_A[int(r_ind-1),i-k]*R_B[int(c_ind)-1,i-k] 
                Wab[i,ind] = Wab[i,ind] - sumA;
                ind = ind+1;

    output_worker = {};
    for i in range (0,np.size(apw)):
        lenA = qA + (kA - 1)*(apw[i]-k);
        lenB = qB + (kB - 1)*(apw[i]-k);
        lenAB = lenA * lenB;
        for j in range (0,lenAB):
            ss = Wab[active_workers[np.size(amw)+i],j];
            ss = np.transpose(np.reshape(ss, (aa*bb, 1)))
            if i==0 and j==0:
                output_worker = ss;
            else:
                output_worker = np.concatenate((output_worker,ss),axis = 0)
        
    start_time = time.time()
                                                              ## Peeling Decoder for all 1's
    AA = csr_matrix(Coding_matrix);
    BB = output_worker;
    zz = (k-np.size(amw))*qA*qB;
    res = np.zeros((zz,aa*bb),dtype=float);
    while csr_matrix.count_nonzero(AA)>0:
        ind1 = AA.sum(axis=1);                                ## Finding the rows with single unknown
        ind3 = np.where(ind1==1)
        ind2 = ind3[0];
        (aa1,bb1) = np.nonzero(AA[ind2,:])
        ij = np.argsort(aa1)
        bb1 = bb1[ij];
        (_, imp_rows) = np.unique(bb1, axis=0,return_index=True);  
        bb1 = bb1[imp_rows];
        res[bb1,:] = BB[ind2[imp_rows],:];                    ## Recovering unknowns
        (ee,ff) = np.nonzero(AA[:,bb1])        
      
        for ii in range(0,np.size(ee)):
            BB[ee[ii],:] = BB[ee[ii],:] - res[bb1[ff[ii]],:];
          
        AA[ee,bb1[ff]] = 0;

    end_time = time.time();
    print("\nDecoding time is %f seconds\n" % (end_time - start_time))        

    unknownsA = np.floor(unknowns/DeltaB)+1;
    unknownsB = np.remainder(unknowns+1,DeltaB);
    unknownsB[unknownsB==0]=DeltaB;
    (xx,yy) = np.shape(res);
    for ii in range (0,xx):
        cur = np.reshape(res[ii,:], (aa, bb));
        r_arr1 = int((unknownsA[ii]-1)*aa)
        r_arr2 = int((unknownsA[ii])*aa)
        c_arr1 = int((unknownsB[ii]-1)*bb)
        c_arr2 = int((unknownsB[ii])*bb)
        final_r[r_arr1:r_arr2,c_arr1:c_arr2] = cur;

    err_ls = 100*np.square(np.linalg.norm(final_r - E)/normE);

    print("Normalized MSE at SNR %s is %s"%(SNR, err_ls))
    comm.Abort()

else:
    SNR = comm.recv(source=0);
    WsendA = comm.recv(source=0)
    lengA = comm.recv(source=0)
    WsendB = comm.recv(source=0)
    lengB = comm.recv(source=0)

    Wab = {};
    ind = 0;
    start_time = time.time()
    for j in range (0,lengA):
        for m in range (0,lengB):
            Wab[ind] = np.matmul(np.transpose(WsendA[j]),WsendB[m]);
            ind = ind + 1;
    end_time = time.time();
    c = end_time - start_time;
    
    Wab1 = {};
    ind = 0;
    for j in range (0,lengA):
        for m in range (0,lengB):
            sig = Wab[ind];
            sig_avg_db = 10*np.log10(np.mean(sig ** 2));
            noise_avg_db = sig_avg_db - SNR;
            var = 10 ** (noise_avg_db / 10);
            Wab1[ind] = sig + np.random.normal(0,np.sqrt(var),np.shape(Wab[ind]));
            ind = ind+1;
    comm.send(Wab1, dest=0)
    print("Computation time for processor %s is %s" % (rank, c))
