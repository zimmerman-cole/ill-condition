function A_E = constr(x)
% input:
%    -- x: 5-dim vector
% output:
%    -- A_E(x): gradient of constraints at x
A_E = ...
      [2.*x(1), 2.*x(2), 2.*x(3), 2.*x(4), 2.*x(5);
       0, x(3), x(2), -5.*x(5), -5.*x(4);
       3.*x(1)^2, 3.*x(2)^2, 0, 0, 0;
       ]
