function B = shiftrow(A,r,t)
[a,b] = size(A);
B = zeros(t,b);
B(r+1:a+r,:) = A;
end