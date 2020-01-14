%   Finding the worst case condition number for the proposed scheme
%
%   We have n workers and s = n - kA*kB  stragglers.
%   The maximum storage for A and B of any worker are gammaA and gammaB.
%   Storage fraction gammaA > 1/kA and gammaB > 1/kB.
%   Matrix A is divided into DeltaA block columns, where kA divides DeltaA.
%   Matrix B is divided into DeltaB block columns, where kB divides DeltaB.
%   DeltaA and DeltaB can be calculated using s, gammaA and gammaB.

%   Set random = 1 if you want an upper bound for the condition number.
%   Coding_matrix gives the system matrix for any set of k = kA * kB workers.
%   worst_condition_number is the maximum condition number in this scheme.
%   Set dist = 'rand' if you want the coefficients from standard normal distribution.
%   One can increase no_trials, which can help to find a better condition number.   


clc
close all
clear

n = 8;
kA = 3;
kB = 2;
gammaA = 3/8;                                  %% gammaA needs to be greater than 1/kA
gammaB = 3/5;                                  %% gammaB needs to be greater than 1/kB
k = kA*kB;      
s = n - k;                                     %% The number of stragglers, s > 1
                                     
random = 1;                                    %% can be 0 or 1.
no_trials = 20;                                %% number of trials, if random = 1 
dist = 'unif';                                 %% distribution, 'rand' or 'unif' 

if (gammaA <= 1/kA), error('Storage for A does not satify the constraint.'); end
if (gammaB <= 1/kB), error('Storage for B does not satify the constraint.'); end

if random == 1
    [R_A, R_B,cond_min] = matrix_matrix_best_mat(n,kA,kB,gammaB,no_trials,dist);
    predicted_upper_bound = cond_min;
else
    R_A = ones(kA,s);
    R_B = ones(kB,s);
end

if random ==1
    M1 = ['The predicted upper bound is ', num2str(predicted_upper_bound),'.'];
    fprintf('\n'); disp(M1);
end

if gammaA <= 1/kA 
    error('Storage fraction for A needs to be greater than 1/kA'); 
elseif gammaB <= 1/kB 
    error('Storage fraction for B needs to be greater than 1/kB'); 
end

DeltaA = round((s-1)*(kA-1)/(gammaA - 1/kA));
DeltaB = round((s-1)*(kB-1)/(gammaB - 1/kB));
while(rem(DeltaA,kA)~=0)    
    DeltaA = DeltaA + 1;
end
qA = DeltaA/kA;
while(rem(DeltaB,kB)~=0)    
    DeltaB = DeltaB + 1;
end
qB = DeltaB/kB;

M2 = ['The value of Delta for A is ', num2str(DeltaA),'.'];
fprintf('\n'); disp(M2);
M2 = ['The value of Delta for B is ', num2str(DeltaB),'.'];
fprintf('\n'); disp(M2);

for i=1:kA
    P = zeros(qA,DeltaA);
    P(1:qA,(i-1)*qA+1:(i)*qA)=eye(qA);
    for j = 1:kB
        Wa{(i-1)*kB+j}=P;     
    end
end
for i=1:kB
    P = zeros(qB,DeltaB);
    P(1:qB,(i-1)*qB+1:(i)*qB)=eye(qB);
    for j = 1:kA
        Wb{(j-1)*kB+i}=P;     
    end
end

zerA = 2*DeltaA;
for i=k+1:n
    Wa{i} = zeros(zerA,DeltaA);
    P = [];
    for j=1:kA
        P = shiftrow(R_A(j,i-k)*Wa{(j-1)*kB+1},(j-1)*(i-k-1),zerA);
        Wa{i} = Wa{i} + P;                        %% Coding Matrix for Parity Workers for A
    end
    Wa{i}(~any(Wa{i},2),:)=[];
end

zerB = 2*DeltaB;
for i=k+1:n
    Wb{i} = zeros(zerB,DeltaB);
    P = [];
    for j=1:kB
        P = shiftrow(R_B(j,i-k)*Wb{j},(j-1)*(i-k-1),zerB);
        Wb{i} = Wb{i} + P;                        %% Coding Matrix for Parity Workers for B
    end
    Wb{i}(~any(Wb{i},2),:)=[];
end

for i=1:n
    Wab{i}=[];
    [len_a,~]=size(Wa{i});
    for j=1:len_a
        [len_b,~]=size(Wb{i});
        for m = 1:len_b
            Wab{i}=[Wab{i};kron(Wa{i}(j,:),Wb{i}(m,:))];
        end
    end
end

workers = 1:n;
Choice_of_workers = combnk(workers,k);          %% Different sets of workers
[total_no_choices,~] = size(Choice_of_workers);     
cond_no = zeros(total_no_choices,1);

for i = 1:total_no_choices
    Coding_matrix = [];
    for j = 1:k
        Coding_matrix = [Coding_matrix ; Wab{Choice_of_workers(i,j)}];
    end
    cond_no(i) = cond(Coding_matrix);
end

worst_condition_number = max(cond_no);
pos = find(cond_no==worst_condition_number);
worst_choice_of_workers = Choice_of_workers(pos,:);

M3 = ['The worst case condition number is ', num2str(worst_condition_number),'.'];
fprintf('\n'); 
disp(M3);
M4 = ['The worst case includes workers ', num2str(worst_choice_of_workers),'.'];
fprintf('\n'); 
disp(M4);
fprintf('\n')
