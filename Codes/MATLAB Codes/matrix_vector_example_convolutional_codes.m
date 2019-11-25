%   Finding the worst case error percentage for our proposed scheme
%   Having a random matrix A of size (r,t) and a vector x of length t.
%   We have n workers, s stragglers.
%   Storage fraction gamma > 1/k ; where k = n - s; and find Delta.
%   Matrix A is divided into Delta block columns, where k divides Delta.
%   One can vary SNR to check the scenario at different noise levels.
%   Set random = 1 for an upper bound for the condition number.
%   Set worst_case = 1 to find the error in the worst_case scenario.
%   worst_condition_number is the maximum condition number in this scheme.
%   Set dist = 'rand' if you want the coefficients from standard normal distribution.
%   One can increase no_trials, which can help to find a better condition number.  

clc
close all
clear

n = 10;
s = 3;                                     %% set straggler number, s > 1
k = n - s;
gamma = 1/5;                               %% needs to be greater than 1/k
Delta = round((s-1)*(k-1)/(gamma - 1/k));
while(rem(Delta,k)~=0)
    Delta = Delta + 1;
end
q = Delta/k;

r = 6300;                                      %% Needs to be a multiple of Delta
t = 2000;
A = randn(t,r);
x = randn(t,1);
b = A'*x;
SNR = 50;
random = 1;                                    %% set 1 to choose random coefficients.
peeling = 0;
if random == 0
    peeling = 1;                               %% Peeling decoder for all 1's case
end
no_trials = 20;                                %% number of trials, if random = 1
dist = 'rand';                                 %% distribution, 'rand' or 'unif'
worst_case = 1;                                %% set 1 to find the worst case error.
if worst_case ~= 1
    all_workers = randperm(n);              
    active_workers = all_workers(1:k);         %% any k out of 1 to n
    active_workers = sort(active_workers);
end

if random == 1
    [best_matrix,predicted_upper_bound] = matrix_vector_best_mat(n,s,no_trials,dist);
else
    best_matrix = ones(k,s);
end
full_mat = [eye(k) best_matrix];

if worst_case == 1
    for i=1:k
        P = zeros(q,Delta);
        P(1:q,(i-1)*q+1:(i)*q)=eye(q);
        W{i}=P;                                     %% Matrix for Message Workers
    end
    
    zer = 2*Delta;
    for i=k+1:n
        W{i} = zeros(zer,Delta);
        P = [];
        for j=1:k
            P = shiftrow(full_mat(j,i)*W{j},(j-1)*(i-k-1),zer);
            W{i} = W{i} + P;                        %% Matrix for Parity Workers
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
    active_workers = Choice_of_workers(pos,:);
    if random ==1
        fprintf('\n'); 
        disp(['The upper bound of the condition number is ', num2str(predicted_upper_bound),'.']);
    end
    fprintf('\n'); 
    disp(['The worst case condition number is ', num2str(worst_condition_number),'.']);
    fprintf('\n'); 
    disp(['The worst case includes workers ', num2str(active_workers),'.']);
end


%%%%% Encoding Starts in the Master Node  %%%%%%

for i=1:k
    W1{i} = A(:,(i-1)*r/k+1:i*r/k);             %% Diving Matrix A in k blocks
end

c = r/Delta;
for i=1:k
    for j =1:q
        W2{i}{j} = W1{i}(:,(j-1)*c+1:j*c);      %% Message Workers' Assignments
    end
end

for i = k+1:n
    len = q + (k-1)*(i-k-1);
    for j =1:len
        sumA = zeros(t,c);
        for rr = 1:k
            if j >= (i-k-1)*(rr-1)+1 && j <= (i-k-1)*(rr-1)+q
                sumA  = sumA + W2{rr}{j-(i-k-1)*(rr-1)}* full_mat(rr,i);
            end
        end
        W2{i}{j} = sumA;                         %% Parity Workers' Assignments
    end
end

%%%%% Computation starts in the worker nodes  %%%%%%

for i = 1:n
    [~,len]=size(W2{i});
    for j=1:len
        W3{i}{j} = awgn(W2{i}{j}'*x,SNR);        %% Worker Computation and Noise
    end
end

for i=1:k
    Coding_matrix = zeros(q,Delta);
    Coding_matrix(1:q,(i-1)*q+1:(i)*q)=eye(q);
    W{i}=Coding_matrix;
end

zer = 2*q*k;
for i=k+1:n
    W{i} = zeros(zer,Delta);
    P = [];
    for j=1:k
        P = shiftrow(full_mat(j,i)*W{j},(j-1)*(i-k-1),zer);
        W{i} = W{i} + P;
    end
    W{i}(~any(W{i},2),:)=[];
end

%%%%%%%% Decoding Starts in Master Node %%%%%

tic

amw = active_workers(active_workers<=k);
apw = active_workers(active_workers>k);

res = zeros(c*q,k);
for i = 1:length(amw)
    for j = 1:q
        res((j-1)*c+1:j*c,amw(i)) = W3{amw(i)}{j};      %% getting the values from the systematic part
    end
end

for i = k+1:n
    if ~ismember(i,apw), continue; end
    len = q + (k-1)*(i-k-1);
    for j =1:len
        sumA = zeros(c,1);
        for rr = 1:k
            if ~ismember(rr,amw), continue; end
            if j >= (i-k-1)*(rr-1)+1 && j <= (i-k-1)*(rr-1)+q
                sumA  = sumA + W3{rr}{j-(i-k-1)*(rr-1)}* full_mat(rr,i);
            end
        end
        W3{i}{j} = W3{i}{j} - sumA;
    end
end

Coding_matrix = [];
for j = 1:length(apw)
    Coding_matrix = [Coding_matrix ; W{active_workers(length(amw)+j)}];
end

[xx,yy] = size(Coding_matrix);
if xx>0
    for i =1:k
        if ismember(i,amw)
            Coding_matrix(:,(i-1)*q+1:i*q)=0;
        end
    end
end
output_worker = [];
for i=1:length(apw)
    [~,len]=size(W3{active_workers(length(amw)+i)});
    for j=1:len
        output_worker = [output_worker;W3{active_workers(length(amw)+i)}{j}'];
    end
end
fprintf('\n');

if peeling == 0                                       %% LS Decoding for random convolutional coding
    Coding_matrix(:,~any(Coding_matrix,1))=[];
    res2 = sparse(Coding_matrix)\output_worker;       %% results from the parity parts using LS
    res_trans = res2';
    coded_res = res_trans(:);
    
    count = 0;
    for i = 1:k
        if ~ismember(i,amw)
            count = count + 1;
            res(:,i) = coded_res((count-1)*c*q+1:count*c*q);
        end
    end
    
else                            %% Peeling Decoder for all 1's
    res2 = (reshape(res(:),[c,Delta]))';
    AA = Coding_matrix;
    BB = output_worker;
    zer_rows = find(all(AA==0,2));
    AA(zer_rows,:)=[];
    BB(zer_rows,:)=[];
    
    while any(AA(:))
        [~,imp_rows] = unique(AA,'rows');       %% removing the unique rows
        imp_rows = sort(imp_rows);
        AA = AA(imp_rows,:);
        BB = BB(imp_rows,:);
        
        [u,v] = size(AA);
        ind1 = sum(AA~=0,2);                    %% Finding the rows with single unknown
        ind2 = find(ind1==1);
        [aa,bb] = find(AA(ind2,:)~=0);
        [aa,ij] = sort(aa);
        bb = bb(ij);
        res2(bb,:) = BB(ind2,:);                %% Recovering unknowns
        [ee,ff] = find(AA(:,bb)~=0);
        for ii = 1:length(ee)
            BB(ee(ii),:) = BB(ee(ii),:)- res2(bb(ff(ii)),:);
        end
        
        AA(ee,bb(ff))=0;
        AA(ind2,:)=[];                          %% Removing the already used rows
        BB(ind2,:)=[];
    end
    res = res2';
end

final_result = res(:);
decoding_time = toc;
disp(['The decoding time is ', num2str(decoding_time),'seconds.']);
err_ls = 100*(norm(final_result-b,'fro')/norm(b,'fro'))^2;
fprintf('\n');
disp(['The error is ', num2str(err_ls),' % .']);
fprintf('\n');
