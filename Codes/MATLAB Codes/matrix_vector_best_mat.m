function [R,best_cond_min] = matrix_vector_best_mat(n,s,no_trials,dist)

%   best_mat(n,s) finds a random matrix to optimize the worst case
%   condition number for n workers and s stragglers.
%
%   best_mat(n,s,no_trials) tries to optimize by running the simulation for
%   no_trials times.
%   
%   best_mat(n,s,no_trials,dist) tries to optimize by running the
%   simulation for no_trials times drawing the coefficients from any
%   specific distribution where dist can be 'rand' or 'unif'.
%   
%   The function returns R as the best random matrix and best_cond_min as
%   the minimum possible worst case condition number.
%    
%   We follow the methods described in the paper- 
%   Houcem Gazzah, Phillip A Regalia, and J-P Delmas, "Asymptotic eigenvalue 
%   distribution of block toeplitz matrices and application to blind simo 
%   channel identification. IEEE Trans. on Info. Th., 47(3):1243–1251, 2001.
% 

if (nargin < 3 || isempty(no_trials)), no_trials = 50;    end
if (nargin < 4 || isempty(dist)), dist = 'rand';    end

samples_in_omega = 200;
k = n - s;
message = eye(k);
condition_number = zeros(no_trials,1);

for trial=1:no_trials
    if strcmp(dist,'unif')
        parity{trial} = unifrnd(-1,1,k,s);
    else
        parity{trial} = randn(k,s);
    end
    
    ind = 0;
    for w=-pi:pi/samples_in_omega:pi
        imag = 1i;
        for i = 1:s
            omega(:,i) = ((exp(-imag*w))^(i-1)).^(0:k-1);
        end
        Generator_mat = [message parity{trial}.*omega];
        
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
R = parity{position};

