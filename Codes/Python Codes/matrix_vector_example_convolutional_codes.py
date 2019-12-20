"""
Having a random matrix A of size (r,t) and a vector x of length t.
Finding the worst case error percentage for our proposed scheme to find A'x
We have n workers, s stragglers.
Storage fraction gamma > 1/k ; where k = n - s; and find Delta.
Matrix A is divided into Delta block columns, where k divides Delta.
One can vary SNR to check the scenario at different noise levels.
Set random = 1 for an upper bound for the condition number.
Set worst_case = 1 to find the error in the worst_case scenario.
Worst_condition_number is the maximum condition number in this scheme.
One can increase no_trials, which can help to find a better condition number.  
"""

import numpy as np
from scipy import linalg
import itertools as it
import time
from scipy.sparse import csr_matrix


def shiftrow(A,r,t):   
    (a,b) = np.shape(A) ;
    B = np.zeros((t,b),dtype = float) ;
    B[r:a+r,:] = A ;
    return B;

def matrix_vector_best_mat(n,s,no_trials):
    samples_in_omega = 400;
    k = n - s ;
    message = np.identity(k,dtype = float);
    condition_number = np.zeros(no_trials,dtype = float);
    parity = {} ;
    workers = list(range(n));
    Choice_of_workers = list(it.combinations(workers,k));
    size_total = np.shape(Choice_of_workers);            
    total_no_choices = size_total[0];
    min_eigenvalue = np.zeros(total_no_choices*samples_in_omega);
    max_eigenvalue = np.zeros(total_no_choices*samples_in_omega);
    
    for t in range (0,no_trials):
        parity[t] = np.random.rand(k,s);
        ind = 0;
        min_eigenvalue = np.zeros(total_no_choices*samples_in_omega);
        max_eigenvalue = np.zeros(total_no_choices*samples_in_omega);
        w = np.zeros(2*samples_in_omega,dtype = float);
        for i in range (0,samples_in_omega):
            w[i] = -np.pi + i*2*np.pi/samples_in_omega;
        zz = samples_in_omega;
        for z in range (0,zz):
            imag = 1j;
            omega = np.zeros((k,s),dtype = complex);
            for i in range (0,s):
                omega[:,i] = np.power(np.exp(-imag*w[z])**i,list(range(k)))
            Generator_mat = np.concatenate((message, np.multiply(parity[t],omega)), axis = 1)
            for i in range (0,total_no_choices):
                Coding_matrix = [];
                kk = list(Choice_of_workers[i]);
                Coding_matrix = Generator_mat[:,kk];
                Coding_matrixT = np.transpose(Coding_matrix);
                D = np.matmul(np.conjugate(Coding_matrixT),Coding_matrix);
                eigenvalues = linalg.eigvals(D);
                eigenvalues = np.real(eigenvalues);
                min_eigenvalue[ind] = np.min(eigenvalues);
                max_eigenvalue[ind] = np.max(eigenvalues);
                ind = ind + 1;
        condition_number[t] = np.sqrt(np.max(max_eigenvalue)/np.min(min_eigenvalue)) 
    best_cond_min = np.min(condition_number);
    position =   np.argmin(condition_number);
    R = parity[position];
    return R,best_cond_min        
            

n = 10 ;
s = 3 ;                                             ## set straggler number, s > 1
k = n - s ;
gamma = 1/5;                                        ## needs to be greater than 1/k
SNR = 80; 
Delta = int(np.round((s-1)*(k-1)/(gamma-1/k))); 
while Delta % k != 0:
    Delta = Delta+1 ;
q = int(Delta/k) ;                                 ## Needs to be a multiple of Delta
print('\nThe value of Delta is %s' % Delta)
r = 6300 ;
t = 1000 ;
mu = 0;
sigma = 1;
A = np.random.normal(mu, sigma, [t,r]);
x = np.random.normal(mu, sigma, [t,1]);
b = np.matmul(np.transpose(A),x);
random = 1;                                        ## set 1 to choose random coefficients.
no_trials = 25;                                     ## number of trials, if random = 1
worst_case = 1;                                    ## set 1 to find the worst case error

peeling = 0;
if random == 0:
    peeling = 1;                                  ## Peeling decoder for all 1's case
    

if worst_case !=1:
    all_workers = np.random.permutation(n);
    active_workers = all_workers[0:k];
    active_workers.sort() ;

if random == 1:
    (best_matrix,predicted_upper_bound) = matrix_vector_best_mat(n,s,no_trials);
else:
    best_matrix = np.ones((k,s),dtype = float);

full_mat = np.concatenate((np.identity(k,dtype = float),best_matrix), axis = 1) ;
W = {} ;
if worst_case == 1:
    for i in range (0,k):
        P = np.zeros((q,Delta),dtype=float);
        P[0:q,i*q:(i+1)*q] = np.identity(q,dtype = float);
        W[i] = P;                               ## Matrix for Message Workers

    zer = 2*Delta ;
    for i in range (k,n):
        W[i] = np.zeros((zer,Delta),dtype=float);
        P = [];
        for j in range (0,k):
            P = shiftrow(full_mat[j,i]*W[j],j*(i-k),zer);
            W[i] = W[i] + P;                    ## Matrix for Parity Workers
        inter = W[i];
        W[i] = inter[~np.all(inter == 0, axis=1)];
    workers = list(range(n));
    Choice_of_workers = list(it.combinations(workers,k));          ## Different sets of workers
    size_total = np.shape(Choice_of_workers);            
    total_no_choices = size_total[0];
    cond_no = np.zeros(total_no_choices,dtype = float);    
    for i in range (0, total_no_choices):
        for j in range (0,k):
            kk = list(Choice_of_workers[i]);
            if j==0:
                Coding_matrix = W[kk[j]];
            else:
                Coding_matrix = np.concatenate((Coding_matrix, W[kk[j]]), axis = 0);
        cond_no[i] = np.linalg.cond(Coding_matrix);    
    worst_condition_number = np.max(cond_no);
    pos =   np.argmax(cond_no);
    active_workers = list(Choice_of_workers[pos]);       
    if random == 1:
        print('\nUpper Bound of the condition number is %s' % predicted_upper_bound);
    print('\nWorst case condition number is %s' % worst_condition_number)
    print('\nActive workers are %s' %active_workers)          

W1 = {} ;
for i in range (0,k):
    W1[i] = A[:,int(i*r/k):int((i+1)*r/k)] ;  
del A
W2 = {} ;
c = int(r/Delta);
for i in range (0,k):
    for j in range (0,q):
        W2[i,j] = W1[i][:,j*c:(j+1)*c] ;                ## Message Workers' Assignments
del W1
for i in range (k,n):
    len = q + (k-1)*(i-k);
    for j in range (0,len):
        sumA = np.zeros((t,c),dtype = float);
        for rr in range (0,k):
            if j >= (i-k)*rr and j <= (i-k)*rr+q-1:
                sumA = sumA + W2[rr,j-(i-k)*rr]*full_mat[rr,i];
                
        W2[i,j] = sumA;                                 ##  Parity Workers' Assignments
for i in range (0,n):
    leng = q if i < k else q + (k-1)*(i-k) ;
Wsend = {};
for j in range (0,leng):
	 Wsend[j] = W2[i,j];
W3 = {} ;        
for i in range (0,n):
    len = q if i < k else q + (k-1)*(i-k) ;
    for j in range (0,len):
        W3[i,j] = np.matmul(np.transpose(W2[i,j]),x);
        sig = W3[i,j];
        sig_avg_db = 10*np.log10(np.mean(sig ** 2));
        noise_avg_db = sig_avg_db - SNR;
        var = 10 ** (noise_avg_db / 10);
        W3[i,j] = W3[i,j] + np.random.normal(0,np.sqrt(var),np.shape(W3[i,j]));
        
W = {};
for i in range (0,k):
    Coding_matrix = np.zeros((q,Delta),dtype = float);
    Coding_matrix[0:q,i*q:(i+1)*q] = np.identity(q,dtype = float);
    W[i] = Coding_matrix ; 
    
zer = 2*q*k;
for i in range (k,n):
    W[i] = np.zeros((zer,Delta));
    P = [];
    for j in range (0,k):
        P = shiftrow(full_mat[j,i]*W[j],j*(i-k),zer);
        W[i] = W[i] + P ;
    inter = W[i];
    W[i] = inter[~np.all(inter == 0, axis=1)]   


start_time = time.time()
ss = np.array(active_workers)
amw = ss[ss<k];
apw = ss[ss>=k];

res = np.zeros((c*q,k),dtype = float);
for i in range(0,np.size(amw)):
    for j in range(0,q):
        dd = W3[amw[i],j];
        res[j*c:(j+1)*c,amw[i]]  = dd.ravel()           ## getting the values from the systematic part

for i in range(k,n):
    if i in apw:
        leng = q + (k-1)*(i-k);
        for j in range (0,leng):
            sumA = np.zeros((c,1),dtype = float);
            for rr in range(0,k):
                if rr in amw:
                    if j >= (i-k)*rr and j <= (i-k)*rr+q-1:
                        sumA = sumA + full_mat[rr,i]*W3[rr,j-(i-k)*rr];
            W3[i,j] = W3[i,j]-sumA

           
if np.size(apw)>0:
    Coding_matrix = W[active_workers[np.size(amw)]];
    for j in range (1,np.size(apw)):
        Coding_matrix = np.concatenate((Coding_matrix, W[active_workers[np.size(amw)+j]]), axis = 0) ;

[xx,yy]=np.shape(Coding_matrix)
if xx>0:
    for i in range (0,k):
        if i in amw:
            Coding_matrix[:,i*q:(i+1)*q] = 0;

for i in range (0,np.size(apw)):
    leng = q + (k-1)*(active_workers[np.size(amw)+i]-k) ;
    for j in range (0,leng):
        if i==0 and j==0:
            output_worker = np.transpose(W3[active_workers[np.size(amw)],0]);
        else:
            output_worker = np.concatenate((output_worker, np.transpose(W3[active_workers[np.size(amw)+i],j])), axis = 0) ;       
          
if peeling == 0:                               ## LS Decoding for random convolutional coding
    inter = np.transpose(Coding_matrix);
    Coding_matrix = np.transpose(inter[~np.all(inter == 0, axis=1)])  
    CC = np.linalg.inv(np.matmul(np.transpose(Coding_matrix),Coding_matrix))
    res_p = np.matmul(CC,np.matmul(np.transpose(Coding_matrix),output_worker))
    
    atw = list(range(k));
    aow = [i for i in atw if i not in amw];
    res_p = np.reshape(res_p, (q*c*np.size(aow), 1));

    for j in range(0,np.size(apw)):
        res[:,aow[j]] = res_p[j*c*q:(j+1)*c*q].ravel();

else:                                   ## Peeling Decoder for all 1's
    AA = csr_matrix(Coding_matrix);
    BB = output_worker;
    res2 = np.transpose(res);
    res2 = (np.reshape(res2, (Delta,c)));
    while csr_matrix.count_nonzero(AA)>0:
        ind1 = AA.sum(axis=1);                      ## Finding the rows with single unknown
        ind3 = np.where(ind1==1)
        ind2 = ind3[0];
        (aa,bb) = np.nonzero(AA[ind2,:])
        ij = np.argsort(aa)
        bb = bb[ij];
        (_, imp_rows) = np.unique(bb, axis=0,return_index=True);  
        bb = bb[imp_rows];
        res2[bb,:] = BB[ind2[imp_rows],:];                    ## Recovering unknowns
        (ee,ff) = np.nonzero(AA[:,bb])        
        for ii in range(0,np.size(ee)):
            BB[ee[ii],:] = BB[ee[ii],:] - res2[bb[ff[ii]],:];
            
        AA[ee,bb[ff]] = 0;

    res = np.transpose(res2);

res = np.transpose(res);
final_result = np.reshape(res, (r, 1));
end_time = time.time();

print('\nTime needed is %s' % (end_time - start_time))
err_ls = 100*np.square(np.linalg.norm(final_result - b,'fro')/np.linalg.norm(b,'fro'));
print('\nError percentage is %s' % err_ls)
print('\n')
