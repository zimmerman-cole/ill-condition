
function [y] = ipm_example()
% total variables (y and s)
n = 2;

% objective
f = @(y) y^2;
% constraint
c = @(y,s) 9 - 2*y -s;
% gradient of objective
fx = @(y) [2*y; 0];
% gradient of constraint - Jacobian
cx = [-2; -1];
% second gradient of objective - Hessian
fxx = [2 0; 0 0];
% second gradient of constraint - Hessian
cxx = [0 0; 0 0];

% Initial guess values
y = 3;
s = 9-2*y;
lam = 1;
x = [y s]';

% Solver tuning
mu = 10;
alpha = 0.5;

% Dual variables
z = mu ./ x';
X = diag(x);
Z = diag(z);
e = ones(n,1);

% Hessian of the Lagrangian
W = fxx + lam * cxx;

% Iterations
disp(' Objective         y         s       lam')
for i = 1:10,
    disp(i)
    % Solve A*d = b for d (search direction)
    A = [W   cx        -eye(n);
        cx'  0          zeros(1,n);
        Z    zeros(n,1) X];
    b = -[fx(y) + cx * lam - z';
        c(y,s);
        X*Z*e-mu*e];

    % search direction
    d = inv(A) * b;

    % update values
    y    = y    + alpha * d(1);
    s    = s    + alpha * d(2);
    lam  = lam  + alpha * d(3);
    z(1) = z(1) + alpha * d(4);
    z(2) = z(2) + alpha * d(5);

    % print summary
    disp([f(y),y,s,lam])

    % update x, X, Z
    x = [y s]';
    X = diag(x);
    Z = diag(z);

    % lower mu
    mu = mu / 10;

    % take full steps on future iterations
    alpha = 1.0;
end

disp(y)
