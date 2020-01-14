%   Finding the worst case condition number for the proposed scheme
%
%   We have n workers, s stragglers and the storage fraction is gamma.
%   Storage fraction gamma needs to be greater than 1/k ; where k = n - s.
%   Delta can be calculated using s, k and gamma.
%   Matrix A is divided into Delta block columns, where k divides Delta.
%   W{i} denotes the coding matrix for i-th worker, i = 1, 2, 3, ..., n.
%   W{1}, W{2}, ..., W{k} are for the message workers.
%   W{k+1}, W{k+2}, ..., W{n} are for the parity workers.
%   Set random = 1 if you want an upper bound for the condition number.
%   If random = 0, then the scheme will set all the coefficients as 1.
%   Coding_matrix gives the system matrix for any set of k workers.
%   worst_condition_number is the maximum condition number in this scheme.
%   Set dist = 'unif' if you want the coefficients as continuous uniform 
%   random numbers in [-1,1].
%
%   One can increase no_trials, which can help to find a better condition 
%   number. One may not always get the exact same value.
%

clc
close all
clear

n = 6;
s = 2;                                         %% set straggler number, s > 1
k = n-s;
gamma = 1/3;                                   %% Gamma needs to be greater than 1/k   
random = 1;                                    %% can be 0 or 1.
no_trials = 50;                                 %% number of trials, if random = 1 
dist = 'rand';                                 %% distribution, 'rand' or 'unif' 

if gamma <= 1/k 
    error('Storage fraction needs to be greater than 1/k'); 
end

if random == 1
    [R,cond_min] = matrix_vector_best_mat(n,s,no_trials,dist);
    predicted_upper_bound = cond_min;
    fprintf('\n'); 
    disp(['The predicted upper bound is ', num2str(predicted_upper_bound),'.']);
else
    R = ones(k,s);
end
full_mat = [eye(k) R];

Delta = round((s-1)*(k-1)/(gamma - 1/k));
while(rem(Delta,k)~=0)    
    Delta = Delta + 1;
end

q = Delta/k;
fprintf('\n'); 
disp(['The value of Delta is ', num2str(Delta),'.']);
    
for i=1:k
    P = zeros(q,Delta);
    P(1:q,(i-1)*q+1:(i)*q)=eye(q);
    W{i}=P;                                     %% Coding Matrix for Message Workers
end

zer = 2*q*k;
for i=k+1:n
    W{i} = zeros(zer,Delta);
    P = [];
    for j=1:k
        P = shiftrow(full_mat(j,i)*W{j},(j-1)*(i-k-1),zer);
        W{i} = W{i} + P;                        %% Coding Matrix for Parity Workers
    end
    W{i}(~any(W{i},2),:)=[];
end

workers = 1:n;
Choice_of_workers = combnk(workers,k);          %% Different sets of workers
[total_no_choices,~] = size(Choice_of_workers);     
cond_no = zeros(total_no_choices,1);

for i = 1:total_no_choices
    Coding_matrix = [];
    for j = 1:k
        Coding_matrix = [Coding_matrix ; W{Choice_of_workers(i,j)}];
    end
    cond_no(i) = cond(Coding_matrix);
end

worst_condition_number = max(cond_no);
pos = find(cond_no==worst_condition_number);
worst_choice_of_workers = Choice_of_workers(pos,:);

fprintf('\n'); 
disp(['The actual worst case condition number is ', num2str(worst_condition_number),'.']);
fprintf('\n'); 
disp(['The worst case includes workers  ', num2str(worst_choice_of_workers),'.']);
fprintf('\n')
