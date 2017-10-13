function [c_E, c_I, A_E, A_I] = constr(x, s)
% input:
%    -- x: 5-dim vector
% output:
%    -- c_E(x): equality constraints at x
%    -- c_I(x): inequality constraints at x
%    -- A_E(x): gradient of equality constraints at x
%    -- A_I(x): gradient of inequality constraints at x
c_E = ...
[
   x(1)^2 + x(2)^2 + x(3)^2 + x(4)^2 + x(5)^2;
   x(2)*x(3) - 5*x(4)*x(5);
   x(1)^3 + x(2)^3 + 1
];

c_I = ...
[
   x(2) - s(1);
   x(3) - s(2)
];

A_E = ...
[
   2.*x(1), 2.*x(2), 2.*x(3), 2.*x(4), 2.*x(5);
   0, x(3), x(2), -5.*x(5), -5.*x(4);
   3.*x(1)^2, 3.*x(2)^2, 0, 0, 0
];

A_I = ...
[
   0, 1., 0, 0, 0;
   0, 0, 1., 0, 0
];
