import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import util, optimize, test_solvers
import time, datetime
import os

## setup problem
# n = 10
# cond_num = 25
# A = util.psd_from_cond(cond_num,n)
# x_true = np.random.randn(n)
# x_0 = np.random.randn(n)
# b = np.dot(A,x_true)

# A = np.random.randn(n,n)
# U,D,V = la.svd(A)
# print(D)
# can have negative singular values, but by convention, positive


## ITERATIVE REFINEMENT TESTING
# t = test_solvers.Tester()
# t.fit(n_sims=10, cond_num=10**5, m=1000, n=1000, p_xax=0, p_comp=0)
# t.gen_data()
# s1 = "GradientDescentSolver"
# s2 = "ConjugateGradientsSolver"
# s3 = "IterativeRefinementGeneralSolver"
# t.test_spsd(s1)
# t.test_spsd(s2)
# t.test_spsd(solver=s3,intermediate_solver="DirectInverseSolver",intermediate_iter=5,intermediate_continuation=True)
# t.test_spsd(solver=s3,intermediate_solver="ConjugateGradientsSolver",intermediate_iter=5,intermediate_continuation=True)
# t.test_spsd(solver=s3,intermediate_solver="GradientDescentSolver",intermediate_iter=5,intermediate_continuation=True)
# t.test_spsd(solver=s3,intermediate_solver="DirectInverseSolver",intermediate_iter=5,intermediate_continuation=False)
# t.test_spsd(solver=s3,intermediate_solver="ConjugateGradientsSolver",intermediate_iter=5,intermediate_continuation=False)
# t.test_spsd(solver=s3,intermediate_solver="GradientDescentSolver",intermediate_iter=5,intermediate_continuation=False)


## SPECTRUM SHAPE TESTING with multiple methods
n = 500
cond_num = 10000
n_sims = 2
eps = 10
t = "hang"
xax = "time"
fig = plt.figure("residuals")
ax = plt.subplot(111)

full_iter = 500
int_iter = 100


for sim in range(n_sims):
    ## generate problem
    if t == "decay":
        A = util.decaying_spd(cond_num=cond_num, n=n, min_sing=None)
    elif t == "hang":
        A = util.decaying_spd(cond_num=cond_num, n=n, min_sing=None)
    else:
        A = util.psd_from_cond(cond_num,n)
    x_true = np.random.randn(n)
    x_0 = np.random.randn(n)
    b = np.dot(A,x_true)

    ## Conjugate Gradient
    cgs = optimize.ConjugateGradientsSolver(A=A,b=b,full_output=True)
    x_cgs,i_cgs,resids_cgs,errs_cgs = cgs.solve(x_0=x_0,x_true=x_true,max_iter=full_iter)
    for x in resids_cgs:
        print(x[1])
    resids_cgs = [x[0] for x in resids_cgs]
    if xax == "time":
        xax_cgs = [x[1] for x in resids_cgs]
    else:
        xax_cgs = range(len(resids_cgs))
    ax.plot(xax_cgs, resids_cgs, label="cgs", color='r', marker='o', markersize=3)

    ## Direct Solver
    dis = optimize.DirectInverseSolver(A=A,b=b,full_output=True)
    x_dis,i_dis,resids_dis,errs_dis = dis.solve(x_0=x_0,x_true=x_true,max_iter=full_iter)
    resids_dis = [x[0] for x in resids_dis]
    if xax == "time":
        xax_dis = [x[1] for x in resids_dis]
    else:
        xax_dis = range(len(resids_dis))
    ax.plot(xax_dis, resids_dis, label="dis", color='b', marker='o', markersize=3)

    ## Decomposition Solver (lu)
    lus = optimize.DecompositionSolver(A=A,b=b,d_type='lu',full_output=True)
    x_lu,i_lu,resids_lu,errs_lu = dis.solve(x_0=x_0,x_true=x_true,max_iter=full_iter)
    resids_lu = [x[0] for x in resids_lu]
    if xax == "time":
        xax_lu = [x[1] for x in resids_lu]
    else:
        xax_lu = range(len(resids_lu))
    ax.plot(xax_lu, resids_lu, label="lu", color='g', marker='o', markersize=3)

    ## Continued Iterative Refinement (with Conjugate Gradient)
    ir_cgs_ct = optimize.IterativeRefinementGeneralSolver(A=A,b=b,full_output=True, \
                                                       intermediate_solver = eval("optimize.ConjugateGradientsSolver"), \
                                                       intermediate_iter = int_iter, \
                                                       intermediate_continuation = True)
    x_ir_cgs_ct,i_ir_cgs_ct,resids_ir_cgs_ct,errs_ir_cgs_ct = ir_cgs_ct.solve(x_0=x_0,x_true=x_true,max_iter=full_iter)
    resids_ir_cgs_ct = [x[0] for x in resids_ir_cgs_ct]
    if xax == "time":
        xax_ir_cgs_ct = [x[1] for x in resids_ir_cgs_ct]
    else:
        xax_ir_cgs_ct = range(len(resids_ir_cgs_ct))
    ax.plot(xax_ir_cgs_ct, resids_ir_cgs_ct, label="ir_cgs_ct", color='firebrick', marker='o', markersize=3)

    ## Continued Iterative Refinement (with Direct Inverse)
    ir_dis_ct = optimize.IterativeRefinementGeneralSolver(A=A,b=b,full_output=True, \
                                                       intermediate_solver = eval("optimize.DirectInverseSolver"), \
                                                       intermediate_iter = int_iter, \
                                                       intermediate_continuation = True)
    x_ir_dis_ct,i_ir_dis_ct,resids_ir_dis_ct,errs_ir_dis_ct = ir_dis_ct.solve(x_0=x_0,x_true=x_true,max_iter=full_iter)
    resids_ir_dis_ct = [x[0] for x in resids_ir_dis_ct]
    if xax == "time":
        xax_ir_dis_ct = [x[1] for x in resids_ir_dis_ct]
    else:
        xax_ir_dis_ct = range(len(resids_ir_dis_ct))
    ax.plot(xax_ir_dis_ct, resids_ir_dis_ct, label="ir_dis_ct", color='navy', marker='o', markersize=3)

    ## Fixed Iterative Refinement (with Conjugate Gradient)
    ir_cgs_fix = optimize.IterativeRefinementGeneralSolver(A=A,b=b,full_output=True, \
                                                       intermediate_solver = eval("optimize.ConjugateGradientsSolver"), \
                                                       intermediate_iter = int_iter, \
                                                       intermediate_continuation = False)
    x_ir_cgs_fix,i_ir_cgs_fix,resids_ir_cgs_fix,errs_ir_cgs_fix = ir_cgs_fix.solve(x_0=x_0,x_true=x_true,max_iter=full_iter, eps=eps)
    resids_ir_cgs_fix = [x[0] for x in resids_ir_cgs_fix]
    if xax == "time":
        xax_ir_cgs_fix = [x[1] for x in resids_ir_cgs_fix]
    else:
        xax_ir_cgs_fix = range(len(resids_ir_cgs_fix))
    ax.plot(xax_ir_cgs_fix, resids_ir_cgs_fix, label="ir_cgs_fix", color='orangered', marker='o', markersize=3)

    ## Fixed Iterative Refinement (with Direct Inverse)
    ir_dis_fix = optimize.IterativeRefinementGeneralSolver(A=A,b=b,full_output=True, \
                                                       intermediate_solver = eval("optimize.DirectInverseSolver"), \
                                                       intermediate_iter = int_iter, \
                                                       intermediate_continuation = False)
    x_ir_dis_fix,i_ir_dis_fix,resids_ir_dis_fix,errs_ir_dis_fix = ir_dis_fix.solve(x_0=x_0,x_true=x_true,max_iter=full_iter, eps=eps)
    resids_ir_dis_fix = [x[0] for x in resids_ir_dis_fix]
    if xax == "time":
        xax_ir_dis_fix = [x[1] for x in resids_ir_dis_fix]
    else:
        xax_ir_dis_fix = range(len(resids_ir_dis_fix))
    ax.plot(xax_ir_dis_fix, resids_ir_dis_fix, label="ir_dis_fix", color='dodgerblue', marker='o', markersize=3)



plt.yscale('log')
plt.xlabel("iteration")
plt.ylabel("||b-Ax||")
plt.title('Solver Tests')
ax.legend(prop={'size':5})
plt.show()
fname = "../test_results/Spectrum/resids_"+t+".png"
fig.savefig(fname)
