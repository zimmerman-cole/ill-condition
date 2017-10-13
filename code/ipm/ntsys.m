function [J, h, p, err] = ntsys(x, s, y, z, mu)
% sets up Jp = h
% ------------------------
% input:
%    -- x: 5-dim vector
% output:
%    -- J: Jacobian of h at x
%    -- h: KKT conditions

%% checking types
disp(class(x))
disp(class(s))
disp(class(y))
disp(class(z))
disp(class(mu))

%% setup quantities
e = [1.;1.];
S = [s(1), 0;
     0, s(2)];
Z = [z(1), 0;
     0, z(2)];
s = reshape(s, [2,1])
y = reshape(y, [3,1])
z = reshape(z, [2,1])

%% compute and setup quantities
[f, g, H] = obj_fun(x);
[c_E, c_I, A_E, A_I] = constr(x, s);

disp(size(g))
disp(size(A_E'))
disp(size(A_I'))

%% setup system
h = ...
[
   g - A_E'*y - A_I'*z;
   S*z - mu*e;
   c_E;
   c_I  % already includes "- s"
];

J = ...
[
   H, zeros(5,2), -A_E', -A_I';
   zeros(2,5), Z, zeros(2,3), S;
   A_E, zeros(3,2), zeros(3,3), zeros(3,2);
   A_I, -eye(2), zeros(2,3), zeros(2);
];

p = J\h;
err = max([norm(h(1:5),2), norm(h(6:7),2), norm(h(8:10),2), norm(h(11:12),2)]);
