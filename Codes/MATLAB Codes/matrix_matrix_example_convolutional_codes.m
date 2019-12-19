%   Finding the worst case error percentage for our proposed scheme for
%   matrix-matrix multiplication.
%   Having two random matrices A and B of size (t,r) and (t,w).
%   We have n workers, s = n - kA*kB stragglers.
%   Storage fraction gammaA > 1/kA and gammaB > 1/kB.
%   Matrix A is divided into DeltaA block columns, where kA divides DeltaA.
%   Matrix B is divided into DeltaB block columns, where kB divides DeltaB.
%   One can vary SNR to check the scenario at different noise levels.
%   Set random = 1 for an upper bound for the condition number.
%   Set worst_case = 1 to find the error in the worst_case scenario.
%   worst_condition_number is the maximum condition number in this scheme.
%   Set dist = 'rand' if you want the coefficients from standard normal distribution.
%   One can increase no_trials, which can help to find a better condition number.

clc
close all
clear

n = 11;
kA = 3;
kB = 3;
k = kA*kB;
s = n - k;                                          %% The number of stragglers, s > 1
gammaA = 2/5;                                       %% gammaA needs to be greater than 1/kA
DeltaA = round((s-1)*(kA-1)/(gammaA - 1/kA));
while(rem(DeltaA,kA)~=0)
    DeltaA = DeltaA + 1;
end

gammaB = 2/5;                                       %% gammaB needs to be greater than 1/kB
DeltaB = round((s-1)*(kB-1)/(gammaB - 1/kB));
while(rem(DeltaB,kB)~=0)
    DeltaB = DeltaB + 1;
end

qA = DeltaA/kA;
qB = DeltaB/kB;
r = 3000;                                       %% r needs to be a multiple of DeltaA
t = 2000;
w = 3000;                                       %% w needs to be a multiple of DeltaB
A = randn(t,r);
B = randn(t,w);
E = A'*B;
SNR = 100;
normE=norm(E,'fro');

random = 1;                                    %% set 1 to choose random coefficients.
no_trials = 25;                                %% number of trials, if random = 1
dist = 'rand';                                 %% distribution, 'rand' or 'unif'
worst_case = 1;                                %% set 1 to find the worst case error.
peeling = 0;
if random == 0 
    peeling = 1;
end
if random == 1
    [R_A,R_B,best_cond_min] = matrix_matrix_best_mat(n,kA,kB,gammaB,no_trials,dist);
else
    R_A = ones(kA,s);
    R_B = ones(kB,s);
end

if worst_case ~= 1
    all_workers = randperm(n);
    active_workers = sort(all_workers(1:k));   %% any k out of 1 to n
    fprintf('\n');
    disp(['Workers ', num2str(active_workers),' are active.']);
end

aa = r/DeltaA;
for i=1:DeltaA
    Wa{i} = A(:,(i-1)*aa+1:i*aa);              %% submatrices of A
end

bb = w/DeltaB;
for i=1:DeltaB
    Wb{i} = B(:,(i-1)*bb+1:i*bb);              %% submatrices of B
end

for i = 1:k
    for j=1:qA
        Wa1{i}{j} = Wa{(ceil(i/kB)-1)*qA+j};   %% Systematic workers' assignments for A
    end
end

for i = k+1:n
    len = qA + (kA-1)*(i-k-1);
    for j =1:len
        sumA = zeros(t,aa);
        for rr = 1:kA
            if j >= (i-k-1)*(rr-1)+1 && j <= (i-k-1)*(rr-1)+qA
                sumA  = sumA + Wa1{kB*(rr-1)+1}{j-(i-k-1)*(rr-1)}*R_A(rr,i-k);
            end
        end
        Wa1{i}{j} = sumA;                      %% Parity Workers' Assignments for A
    end
end

for i=1:kB
    for j=1:qB
        Wbb{i}{j} = Wb{(i-1)*qB+j};
    end
end

for i=1:k
    Wb1{i} = Wbb{rem(i-1,kB)+1};               %% Systematic workers' assignments for B
end

for i = k+1:n
    len = qB + (kB-1)*(i-k-1);
    for j =1:len
        sumB = zeros(t,bb);
        for rr = 1:kB
            if j >= (i-k-1)*(rr-1)+1 && j <= (i-k-1)*(rr-1)+qB
                sumB  = sumB + Wbb{rr}{j-(i-k-1)*(rr-1)}*R_B(rr,i-k);
            end
        end
        Wb1{i}{j} = sumB;                      %% Parity Workers' Assignments for B
    end
end

for i=1:kA
    coding_matrixA = zeros(qA,DeltaA);
    coding_matrixA(1:qA,(i-1)*qA+1:(i)*qA)=eye(qA);
    M1{i}=coding_matrixA;
end
zer = 2*DeltaA;
for i=k+1:n
    M1{i} = zeros(zer,DeltaA);
    coding_matrixA = [];
    for j=1:kA
        coding_matrixA = shiftrow(R_A(j,i-k)*M1{j},(j-1)*(i-k-1),zer);
        M1{i} = M1{i} + coding_matrixA;
    end
    M1{i}(~any(M1{i},2),:)=[];
end

for i=1:n
    if i>k
        M_A{i} = M1{i};
    else
        M_A{i} = M1{ceil(i/kB)};
    end
end

for i=1:kB
    coding_matrixB = zeros(qB,DeltaB);
    coding_matrixB(1:qB,(i-1)*qB+1:(i)*qB)=eye(qB);
    M2{i}=coding_matrixB;
end
zer = 2*DeltaB;
for i=k+1:n
    M2{i} = zeros(zer,DeltaB);
    coding_matrixB = [];
    for j=1:kB
        coding_matrixB = shiftrow(R_B(j,i-k)*M2{j},(j-1)*(i-k-1),zer);
        M2{i} = M2{i} + coding_matrixB;
    end
    M2{i}(~any(M2{i},2),:)=[];
end

for i=1:n
    if i>k
        M_B{i} = M2{i};
    elseif rem(i,kB)==0
        M_B{i} = M2{kB};
    else
        M_B{i} = M2{rem(i,kB)};
    end
end

for i=1:n
    P{i}=[];
    [la1,~]=size(M_A{i});
    for j=1:la1
        [lb1,~]=size(M_B{i});
        for mm =1:lb1
            P{i}=[P{i};kron(M_A{i}(j,:),M_B{i}(mm,:))];        %% Final Coding Matrix
        end
    end
end

for i = 1:n
    ind = 0;
    [~,la1]=size(Wa1{i});
    for j = 1:la1
        [~,lb1]=size(Wb1{i});
        for m = 1:lb1
            ind = ind+1;
            Wab{i}{ind} = awgn(Wa1{i}{j}'*Wb1{i}{m},SNR);
        end
    end
end

if worst_case == 1
    workers = 1:n;
    Choice_of_workers = combnk(workers,k);          %% Different sets of workers
    [total_no_choices,~] = size(Choice_of_workers);
    cond_no = zeros(total_no_choices,1);
    
    for i = 1:total_no_choices
        coding_matrix = [];
        for j = 1:k
            coding_matrix = [coding_matrix ; P{Choice_of_workers(i,j)}];
        end
        cond_no(i) = cond(coding_matrix);
    end
    
    worst_condition_number = max(cond_no);
    pos = find(cond_no==worst_condition_number);
    active_workers = Choice_of_workers(pos,:);
    if random==1
        fprintf('\n');
        disp(['The upper bound of the condition number is ', num2str(best_cond_min),'.']);
    end
    fprintf('\n');
    disp(['The worst case condition number is ', num2str(worst_condition_number),'.']);
    fprintf('\n');
    disp(['The worst case includes workers ', num2str(active_workers),'.']);
end
fprintf('\n')

%%%%%%% Decoding Starts in the Master Node %%%%%%%%
tic

amw = active_workers(active_workers<=k);
apw = active_workers(active_workers>k);

final_r = zeros(r,w);
for ii=1:length(amw)
    jj = amw(ii);
    r_ind = ceil(jj/kB);
    c_ind = rem(jj,kB);
    if c_ind == 0
        c_ind = kB;
    end
    r1 = (r_ind-1)*qA;
    c1 = (c_ind-1)*qB;
    ind = 1;
    for i =1:qA
        for j = 1:qB
            cur = Wab{jj}{ind};
            r_arr = (r1+i-1)*aa+1:(r1+i)*aa;
            c_arr = (c1+j-1)*bb+1:(c1+j)*bb;
            final_r(r_arr,c_arr) = cur;             %% Obtaining Systematic Part Directly
            ind = ind+1;
        end
    end
end

Coding_matrix = [];
for j = 1:length(apw)
    Coding_matrix = [Coding_matrix ; P{active_workers(length(amw)+j)}];
end

[xx,~] = size(Coding_matrix);
if xx>0
    for i =1:k
        if ismember(i,amw)
            r_ind = ceil(i/kB);
            c_ind = rem(i,kB);
            if c_ind == 0, c_ind = kB; end
            for ii = 1:qA
                for jj = 1:qB
                    Coding_matrix(:,((r_ind-1)*qA+(ii-1))*DeltaB+(c_ind-1)*qB+jj)=0;
                end
            end
        end
    end
end

prev_cod = Coding_matrix';
unknowns = find(~all(prev_cod==0,2));
Coding_matrix(:,~any(Coding_matrix,1))=[];

for i = k+1:n
    if ~ismember(i,apw), continue; end
    lenA = qA + (kA-1)*(i-k-1);
    lenB = qB + (kB-1)*(i-k-1);
    ind = 1;
    for j =1:lenA
        for jj =1:lenB
            sumA = zeros(aa,bb);
            for rr = 1:k
                if ~ismember(rr,amw), continue; end
                r_ind = ceil(rr/kB);
                c_ind = rem(rr,kB);
                if c_ind == 0, c_ind = kB; end
                if j >= (i-k-1)*(r_ind-1)+1 && j <= (i-k-1)*(r_ind-1)+qA
                    if jj >= (i-k-1)*(c_ind-1)+1 && jj <= (i-k-1)*(c_ind-1)+qB
                        r1 = j-(i-k-1)*(r_ind-1);
                        c1 = jj-(i-k-1)*(c_ind-1);
                        sumA  = sumA + Wab{rr}{(r1-1)*qB+c1}* R_A(r_ind,i-k)*R_B(c_ind,i-k);
                    end
                end
            end
            Wab{i}{ind} = Wab{i}{ind} - sumA;
            ind = ind + 1;
        end
    end
end


parity_outputs = [];
for i=1:length(apw)
    parity_outputs = [parity_outputs Wab{apw(i)}];
end
[~,len]=size(parity_outputs);
output_worker = (reshape(cell2mat(parity_outputs),[aa*bb,len]))';

zer_rows = find(all(Coding_matrix==0,2));
Coding_matrix(zer_rows,:)=[];
output_worker(zer_rows,:)=[];

if peeling == 0
    res = Coding_matrix\output_worker;          %% Obtaining results from Parity Part using LS
else
    AA = Coding_matrix;
    BB = output_worker;
    res = zeros((k-length(amw))*qA*qB,aa*bb);
    
    while any(AA(:))       
        ind1 = sum(AA,2);
        ind2 = find(ind1==1);
        [aa1,bb1] = find(AA(ind2,:)~=0);
        [aa1,ij] = sort(aa1);
        bb1 = bb1(ij);
        [~,imp_rows] = unique(bb1,'rows');
        bb1 = bb1(imp_rows);
        res(bb1,:) = BB(ind2(imp_rows),:);
        [ee,ff] = find(AA(:,bb1)~=0);
        [~,imp_rows] = unique(ee,'rows');
        BB(ee(imp_rows),:) = BB(ee(imp_rows),:)- res(bb1(ff(imp_rows)),:);
        ee2 = setdiff(1:length(ee),imp_rows);
        
        for ii = 1:length(ee2)
            gg = ee2(ii);
            BB(ee(gg),:) = BB(ee(gg),:)- res(bb1(ff(gg)),:);
        end
        
        AA(ee,bb1(ff))=0;
    end
    
end

unknownsA = ceil(unknowns/DeltaB);
unknownsB = rem(unknowns,DeltaB);
unknownsB(unknownsB==0)=DeltaB;
[xx,~] = size(res);
for ii=1:xx
    cur = reshape(res(ii,:),[aa bb]);
    r_arr = (unknownsA(ii)-1)*aa+1:unknownsA(ii)*aa;
    c_arr = (unknownsB(ii)-1)*bb+1:unknownsB(ii)*bb;
    final_r(r_arr,c_arr) = cur;
end

decoding_time = toc;
disp(['Decoding time is ', num2str(decoding_time),' seconds.']);
fprintf('\n')
err_ls =100*(norm(E-final_r,'fro')/norm(E,'fro'))^2;
disp(['The error is ', num2str(err_ls),' % .']);
fprintf('\n')
