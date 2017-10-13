function A = calca(n)
A = full(gallery('tridiag',n,-1,2,-1));
A(1,1) = 1; 
end