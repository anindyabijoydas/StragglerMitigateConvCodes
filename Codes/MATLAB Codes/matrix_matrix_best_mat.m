function [R_A,R_B,best_cond_min] = matrix_matrix_best_mat(n,kA,kB,gammaB,no_trials,dist)

%   matrix_matrix_best_mat(n,kA,kB,gammaB) finds a random matrix to optimize 
%   the worst case condition number for n workers and s = n - kA*kB
%   stragglers. The maximum storage for B of any worker is gammaB.
%
%   matrix_matrix_best_mat(n,kA,kB,gammaB,no_trials) tries to optimize by 
%   running the simulation for no_trials times.
%   
%   matrix_matrix_best_mat(n,kA,kB,gammaB,no_trials,dist) tries to optimize 
%   by running the simulation for no_trials times drawing the coefficients 
%   from any specific distribution where dist can be 'rand' or 'unif'.
%   
%   The function returns R_A and R_B as the best random matriices and 
%   best_cond_min as the minimum possible worst case condition number.
%
%   We follow the methods described in the paper- 
%   Houcem Gazzah, Phillip A Regalia, and J-P Delmas, "Asymptotic eigenvalue 
%   distribution of block toeplitz matrices and application to blind simo 
%   channel identification. IEEE Trans. on Info. Th., 47(3):1243–1251, 2001.
% 

if (nargin < 4), error('Not enough input parameters.'); end
if (nargin < 5 || isempty(no_trials)), no_trials = 50; end
if (nargin < 6 || isempty(dist)), dist = 'rand';    end
if (gammaB <= 1/kB), error('Storage for matrix B does not satisfy the constraint.'); end

k = kA*kB;
s = n - k;
identity_part = eye(k);
Samples_in_omega = 200;

DeltaB = round((s-1)*(kB-1)/(gammaB - 1/kB));
while(rem(DeltaB,kB)~=0)
    DeltaB = DeltaB + 1;
end
z = DeltaB/kB + (s-1)*(kB-1);

for trial = 1:no_trials
    clear full_mat best_mat_comb omega_portion min_eigenvalue max_eigenvalue

    if strcmp(dist,'unif')
        best_mat_A{trial} = unifrnd(-1,1,kA,s);
        best_mat_B{trial} = unifrnd(-1,1,kB,s);
    else
        best_mat_A{trial} = randn(kA,s);
        best_mat_B{trial} = randn(kB,s);
    end
    U{1} = best_mat_A{trial} ; U{2} = best_mat_B{trial};
    best_mat_comb = khatrirao(U);
    
    ind = 0;
    exponent_vector = [];
    for i = 1:kA
        m = (i-1)*z;
        exponent_vector=[exponent_vector m:m+kB-1];
    end
    
    for w=-pi:pi/Samples_in_omega:pi
        a = 1i;
        for i = k+1:n
            omega_portion(:,i-k) = (exp(-a*w)).^((i-k-1)*exponent_vector);
        end
        Generator_mat = [identity_part best_mat_comb.*omega_portion];
 
        workers = 1:n;
        Choice_of_workers = combnk(workers,k);
        [total_no_choices,~] = size(Choice_of_workers);
        for i = 1:total_no_choices
            Coding_matrix = [];
            for j = 1:k
                Coding_matrix = [Coding_matrix Generator_mat(:,Choice_of_workers(i,j))];
            end
            D = Coding_matrix'*Coding_matrix;
            ind = ind + 1;
            min_eigenvalue(ind) = min(eig(D));
            max_eigenvalue(ind) = max(eig(D));
        end
    end
    condition_number(trial)=sqrt(max(max_eigenvalue)/min(min_eigenvalue));
end
best_cond_min = min(condition_number);
position = condition_number==min(condition_number);

R_A = best_mat_A{position};
R_B = best_mat_B{position};

