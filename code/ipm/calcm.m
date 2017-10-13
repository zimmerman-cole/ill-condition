function M = calcm(n)
M = zeros(n);
for i = 1:n
    for j = 1:n
        M(i,j) = n - max(i,j) + 1;
    end
end
end