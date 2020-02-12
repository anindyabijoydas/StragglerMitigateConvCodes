%   Finding the worst case condition number of Ortho-Poly Code
%   We have n workers and s = n - kA * kB stragglers.
%   Storage fraction gammaA = 1/kA and gammaB = 1/kB.
%
%   This code uses the approach of the following paper-
%
%   M. Fahim and V. R. Cadambe, "Numerically Stable Polynomially Coded Computing," 
%   2019 IEEE International Symposium on Information Theory (ISIT), 
%   Paris, France, 2019, pp. 3017-3021.

clc
close all
clear

n = 18;                             %% Total Workers
kA = 5;
kB = 3;
k = kA*kB;
s = n - k;                          %% Total Stragglers

i = 1:n;
rho = cos((2*i-1)*pi/(2*n));
TA = zeros(n,kA);
TB = zeros(n,kB);

for r = 1:kA
    TA(:,r) = cos((r-1)*acos(rho));
end

for r = 1:kB
    TB(:,r) = cos((r-1)*kA*acos(rho));
end

TA(:,1) = TA(:,1)/sqrt(2);
TB(:,1) = TB(:,1)/sqrt(2);

T = zeros(n,kA*kB);
for i=1:n
    T(i,:) = kron(TA(i,:),TB(i,:));
end

choices = combnk(1:n,kA*kB);
condition_no = zeros(nchoosek(n,k),1);
for kk = 1:length(choices)
    wor = choices(kk,:);
    R = T(wor,:);
    condition_no(kk) = cond(R);   
end
worst_cond_no = max(condition_no);
pos= find(condition_no == worst_cond_no);
worst_choice_of_workers = choices(pos,:);

M1 = ['The worst case condition number is ', num2str(worst_cond_no),'.'];
fprintf('\n'); disp(M1);
M2 = ['The worst case includes workers ', num2str(worst_choice_of_workers),'.'];
fprintf('\n'); disp(M2);
fprintf('\n');
