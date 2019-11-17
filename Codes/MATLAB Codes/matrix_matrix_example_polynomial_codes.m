%   Finding the worst case error percentage for the Reed-Solomon scheme
%   Having two random matrices A and B
%   We have n workers and s = n - kA * kB stragglers.
%   Storage fraction gammaA = 1/kA and gammaB = 1/kB.
%   Set DeltaA = kA and DeltaB = kB.
%   Matrices A and B are divided into DeltaA and DeltaB block columns.
%   We choose n nodes uniformly spaced in [-1,1].
%   Set worst_case = 1 to find the error in the worst_case scenario.
%
%   This code uses the approach of the following paper-
%
%   Qian Yu, Mohammad Maddah-Ali, and Salman Avestimehr. Polynomial codes: 
%   an optimal design for highdimensional coded matrix multiplication. 
%   In Proc. of Advances in Neural Information Processing Systems
%   (NIPS), pages 4403–4413, 2017


clc
close all
clear

n = 12;
kA = 5;
kB = 2;
k = kA*kB;
s = n - k;

r = 2000;                                       %% r needs to be a multiple of kA
t = 3000;
w = 3000;                                       %% w needs to be a multiple of kB
A = randn(t,r);
B = randn(t,w);
E = A'*B;
SNR = 100;
worst_case = 1;                                 %% set 1 to find the worst case error.

node_points = -1 + 2*(1:n)'/n;                  %% Choosing nodes in [-1,1]
Choice_of_workers = combnk(1:n,k);
[total_no_choices,~] = size(Choice_of_workers);
cond_no = zeros(total_no_choices,1);

for i=1:total_no_choices
    dd = Choice_of_workers(i,:);
    nodes = node_points(dd);
    Coding_matrix = zeros(k,k);
    for j=1:k
        Coding_matrix(j,:) = (nodes(j)).^((1:k)-1);
    end
    cond_no(i) = cond(Coding_matrix);
end

if worst_case ~= 1
    all_workers = randperm(n);              
    active_workers = sort(all_workers(1:k));         %% any k out of 1 to n
    nodes = node_points(active_workers);
    M = ['Workers ', num2str(active_workers),' are active.'];
    fprintf('\n'); disp(M);
else
    worst_condition_number = max(cond_no);
    pos = find(cond_no == max(cond_no));
    worst_choice_of_workers = Choice_of_workers(pos,:);
    
    M1 = ['The worst case condition number is ', num2str(worst_condition_number),'.'];
    fprintf('\n'); disp(M1);
    M2 = ['The worst case includes workers ', num2str(worst_choice_of_workers),'.'];
    fprintf('\n'); disp(M2);
    nodes = node_points(worst_choice_of_workers);
end

Coding_matrix = zeros(k,k);
for j=1:k
    Coding_matrix(j,:) = (nodes(j)).^((1:k)-1);
end

Coding_A = Coding_matrix(:,1:kB:k);
for i=1:kA
    W1a{i} = A(:,(i-1)*r/kA+1:i*r/kA);                  %% Dividing A into DeltaA blocks
end

for i=1:k
    W2a{i} = 0*W1a{1};
    for j=1:kA
        W2a{i} = W2a{i} + Coding_A(i,j)*W1a{j};         %% Assignment for A
    end
end

Coding_B = Coding_matrix(:,1:kB);
for i=1:kB
    W1b{i} = B(:,(i-1)*w/kB+1:i*w/kB);                  %% Dividing B into DeltaB blocks
end
for i=1:k
    W2b{i} = 0*W1b{1};
    for j=1:kB
        W2b{i} = W2b{i} + Coding_B(i,j)*W1b{j};         %% Assignment for B
    end
end

%%%%%%%%% Worker Computation Starts %%%%%%%

for i = 1:k
    worker_product{i} = W2a{i}'*W2b{i};                 %% Matrix-matrix Product
end

for i = 1:k
    noisy_product{i} = awgn(worker_product{i},SNR);     %% Adding Noise
end

%%%%%%%%% Decoding Starts %%%%%%%

tic
decoding_mat = inv(Coding_matrix);
[g,h] = size(noisy_product{i});

BB = [];
for i = 1:k
    BB = [BB noisy_product{i}(:)];
end
decoded_blocks = BB*decoding_mat';
ind = 0;
final_res = [];
for i=1:kA
    reshaped_block = [];
    for j =1:kB
        ind = ind+1;
        reshaped_block = [reshaped_block reshape(decoded_blocks(:,ind),[g h])];
    end
    final_res = [final_res; reshaped_block];
end
decoding_time = toc;
M3 = ['Decoding time is ', num2str(decoding_time),' seconds.'];
fprintf('\n'); 
disp(M3);

error_percentage_rs =100*(norm(E-final_res,'fro')/norm(E,'fro'))^2;
M4 = ['The error is ', num2str(error_percentage_rs),' % .'];
fprintf('\n'); 
disp(M4);
fprintf('\n')
