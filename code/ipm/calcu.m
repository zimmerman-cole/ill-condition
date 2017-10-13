function [num2, den] = calcu(i,j)
num2 = (sqrt(2*i-1)* factorial(j-1)^2)^2
den = factorial(i+j-1) * factorial(j-i)
end