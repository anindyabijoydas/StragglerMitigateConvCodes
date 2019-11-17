%   Finding the worst case error percentage for the Reed-Solomon scheme
%   Having a random matrix A and a vector x
%   We have n workers, s stragglers.
%   Storage fraction gamma = 1/k ; where k = n - s; and set Delta = k.
%   Matrix A is divided into Delta block columns.
%   We choose n nodes uniformly spaced in [-1,1], instead of the integers.
%
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
s = 3;                                      
k = n - s;
Delta = k;

r = 9000;                                   %% Needs to be a multiple of Delta
t = 1000;
A = randn(t,r);
x = randn(t,1);
B = A'*x;
SNR = 100;

c = r/Delta;
for i=1:Delta
    W1{i} = A(:,(i-1)*c+1:i*c);             %% Diving Matrix A in Delta blocks
end

node_points = -1 + 2*(1:n)'/n;              %% Choosing Vandermonde nodes
% node_points = 1:n ;                       %% Choosing Vandermonde nodes
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

worst_condition_number = max(cond_no);
pos = find(cond_no == max(cond_no));
worst_choice_of_workers = Choice_of_workers(pos,:);

fprintf('\n'); 
disp(['The worst case condition number is ', num2str(worst_condition_number),'.']);
fprintf('\n'); 
disp(['The worst case includes workers   ', num2str(worst_choice_of_workers),'.']);

nodes = node_points(worst_choice_of_workers);
Coding_matrix = zeros(k,k);
for j=1:k
    Coding_matrix(j,:) = (nodes(j)).^((1:k)-1);
end

for i=1:k
    W2{i} = 0*W1{i};
    for j=1:k
        W2{i} = W2{i} + Coding_matrix(i,j)*W1{j};       %% Assigning jobs to the worst case k workers
    end
end

for i=1:k
    W3(:,i) = W2{i}'*x;                                  %% Workers do their jobs
end

for i=1:k
    W3(:,i) = awgn(W3(:,i),SNR);                        %% Noise is added at workers' outputs
end
fprintf('\n'); 
W3 = W3';

tic
W4 = inv(Coding_matrix)*W3; 
W4 = W4';
final_result = W4(:);
decoding_time = toc;
disp(['Decoding time is ', num2str(decoding_time),' seconds.']);

error_percentage_rs = 100*(norm(final_result-B))^2/(norm(B))^2;

fprintf('\n'); 
disp(['The worst case error is ', num2str(error_percentage_rs),' % .']);
fprintf('\n')
