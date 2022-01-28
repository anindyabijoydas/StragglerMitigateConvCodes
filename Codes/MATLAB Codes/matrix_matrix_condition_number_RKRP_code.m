%   Finding the worst case condition number of RKRP Code
%   We have n workers and s = n - kA * kB stragglers.
%   Storage fraction gammaA = 1/kA and gammaB = 1/kB.
%
%   This code uses the approach of the following paper-
%
%   Subramaniam, Adarsh M., Anoosheh Heidarzadeh, and Krishna R. Narayanan. 
%   "Random khatri-rao-product codes for numerically-stable distributed 
%   matrix multiplication." In 2019 57th Annual Allerton Conference on 
%   Communication, Control, and Computing (Allerton), pp. 253-259, 2019.

clc
close all
clear

n = 24;                            %% Number of total worker nodes
kA = 4;
kB = 5;
k = kA*kB;
s = n - k;                         %% Number of stragglers

choices = combnk(1:n,k);
[u,~] = size(choices);

AX = [];
for m = 1:kB
    AX = [AX eye(kA,kA)];
end
BX = [];
for p = 1:kB
    c = zeros(kB,kA);
    c(p,:) = 1;
    BX = [BX c];
end

no_trials = 10;                                % Number of trials

for trial = 1 : no_trials
    random_matA{trial} = randn(kA,s);          % Random Coefficients for A 
    random_matB{trial} = randn(kB,s);          % Random Coefficients for B

    coding_matrix_A = [AX random_matA{trial}];
    coding_matrix_B = [BX random_matB{trial}];

    coding_matrix_AB = [];
    for p=1:n
        coding_matrix_AB = [coding_matrix_AB kron(coding_matrix_A(:,p),coding_matrix_B(:,p))];
    end

	ond_chk_rs = zeros(u,1);
    for i=1:u
        AB = [];
        for x=1:k
            AB = [AB coding_matrix_AB(:,choices(i,x))];
        end
        cond_chk_rand(i) = cond(AB);
    end

    worst_condition_no(trial) = max(cond_chk_rand);
    pos(trial) = find(cond_chk_rand == worst_condition_no(trial));
    worst_choice_of_workers{trial} = choices(pos(trial),:);
end

worst_cond_no_over_trials = min(worst_condition_no);
pos = find(worst_condition_no == worst_cond_no_over_trials);

M1 = ['The worst case condition number is ', num2str(worst_cond_no_over_trials),'.'];
fprintf('\n'); disp(M1);
M2 = ['The worst case includes workers ', num2str(worst_choice_of_workers{pos}),'.'];
fprintf('\n'); disp(M2);
fprintf('\n');
