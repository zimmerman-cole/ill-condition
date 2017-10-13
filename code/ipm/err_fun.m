function e = err_fun(x, s, y, z, mu)
% input:
%    -- x: 5-dim vector
% output:
%    -- e: error

[J, h, p] = ntsys(x, s, y, z, mu);
e = max(norm(h(1)), norm(h(2)), norm(h(3)), norm(h(4)))
