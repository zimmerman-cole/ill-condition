import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import time, optimize, util
import scipy, bfgs
import scipy.sparse as sps

"""
Temporary holding place for stuff /// stuff that doesn't fit anywhere else.
"""

# || b - Ax ||
def norm_dif(x, *args):
    """
    Return || b - Ax || (Frobenius norm).
    """
    A, b = args
    return la.norm(b - A.dot(x))

def compare_gd():
    """
    Compares all the various implementations of gradient descent we've done.
    """
    best_times = dict()
    best_times['GD'], best_times['Alt'], best_times['Non'], best_times['cl'] = 0, 0, 0, 0
    best_errs = dict()
    best_errs['GD'], best_errs['Alt'], best_errs['Non'], best_errs['cl'] = 0, 0, 0, 0

    gd_e, alt_e, non_e, cl_e = 0, 0, 0, 0

    gd = optimize.GradientDescentSolver(full_output=True)

    for i in range(100):
        A = util.psd_from_cond(cond_num=100, n=100)
        true_x = 4 * np.random.randn(100)
        b = np.dot(A, true_x)

        init_x = np.zeros(100)
        print('Initial error: %f' % norm_dif(init_x, A, b))

        st_time = time.time()
        alt_opt, n_iter, suc, resids = optimize.gradient_descent_alt(A, b, full_output=True, recalc=1)
        alt_time = time.time() - st_time
        alt_err = norm_dif(alt_opt, A, b)
        alt_e += alt_err
        print('Alt took %f seconds' % alt_time)
        print('Alt error: %f (%d iter)' % (alt_err, n_iter))

        st_time = time.time()
        non_opt, n_iter, suc, resids = optimize.gradient_descent_nonsymm(A, b, full_output=True)
        non_time = time.time() - st_time
        non_err = norm_dif(non_opt, A, b)
        non_e += non_err
        print('Non took %f seconds' % non_time)
        print('Non error: %f (%d iter)' % (non_err, n_iter))

        st_time = time.time()
        gd_opt, n_iter, suc, resids = optimize.gradient_descent(A, b, full_output=True)
        gd_time = time.time() - st_time
        gd_err = norm_dif(gd_opt, A, b)
        gd_e += non_e
        print('GD took %f seconds' % gd_time)
        print('GD error: %f (%d iter)' % (gd_err, n_iter))

        st_time = time.time()
        gd.A, gd.b = A, b
        cl_opt, n_iter, resids = gd.solve()
        cl_time = time.time() - st_time
        cl_err = norm_dif(cl_opt, A, b)
        cl_e += cl_err
        print('cl took %f seconds' % cl_time)
        print('cl error: %f (%d iter)' % (cl_err, n_iter))

        errs = sorted([(gd_err, 'GD'), (non_err, 'Non'), (alt_err, 'Alt'), (cl_err, 'cl')])
        times = sorted([(gd_time, 'GD'), (non_time, 'Non'), (alt_time, 'Alt'), (cl_time, 'cl')])

        best_errs[errs[0][1]] += 1
        best_times[times[0][1]] += 1


    print('Errors:')
    print('GD: %d, Alt: %d, Non: %d, cl: %d' % (best_errs['GD'], best_errs['Alt'], best_errs['Non'], best_errs['cl']))
    gd_e, alt_e, non_e, cl_e = gd_e / 100.0, alt_e / 100.0, non_e / 100.0, cl_e / 100.0
    print('Averages:')
    print('GD: %f, Alt: %f, Non: %f, cl: %f' % (gd_e, alt_e, non_e, gd_e))

    print('Times:')
    print('GD: %d, Alt: %d, Non: %d, cl: %d' % (best_times['GD'], best_times['Alt'], best_times['Non'], best_times['cl']))

def plot_iter_refine(cond_num=1000, m=100, n=100):
    """
    Plot residuals vs time for iterative refinement w/ decaying epsilon.
    """
    A = util.mat_from_cond(cond_num=cond_num, m=m, n=n)
    true_x = 4 * np.random.randn(n)
    b = np.dot(A, true_x)

    irs = optimize.IterativeRefinementSolver(A=A, b=b, full_output=True)

    x = np.zeros(n)
    print('Initial residual norm: %f' % norm_dif(x, A, b))

    xopt, n_iter, suc, resids = optimize.iter_refinement_eps(A, b, full_output=True)
    print('Method error: %f (%d iter)' % (norm_dif(xopt, A, b), n_iter))

    xopt, n_iter, c_resids = irs.solve()
    print('Class error: %f (%d iter)' % (norm_dif(xopt, A, b), n_iter))

    plt.plot(resids.keys(), resids.values(), marker='o')
    #plt.plot([i for i in range(len(resids))], resids.values(), marker='o')             # VERSUS ITER NUMBER
    #plt.plot([i for i in range(len(c_resids))], [i[0] for i in c_resids], marker='o')  #   " "
    plt.plot([i[1] for i in c_resids], [i[0] for i in c_resids], marker='o')
    plt.yscale('log')
    plt.ylabel('Residual ||Ax - b||')
    plt.xlabel('Time')
    plt.title('dim(A): %d x %d  cond(A): %d' % (m, n, cond_num))
    plt.legend(['Method', 'Class'])
    plt.show()

    return resids

def compare_cg(plot=True):
    """
    Compare optimize.conjugate_gradient_ideal, optimize.conjugate_gradient_psd
        and optimize.ConjugateGradientsSolver.
    """

    cgs = optimize.ConjugateGradientsSolver(full_output=True)

    psd_avg_err = 0
    psd_avg_time = 0
    id_avg_err = 0
    id_avg_time = 0
    cl_avg_err = 0
    cl_avg_time = 0

    for i in range(100):
        A = util.psd_from_cond(cond_num=1000, n=100)
        x_true = 4 * np.random.randn(100)
        b = np.dot(A, x_true)

        if plot: print('Initial error: %f' % norm_dif(np.zeros(100), A, b))

        st_time = time.time()
        xopt, n_iter, suc, id_resids = optimize.conjugate_gradient_ideal(A, b, full_output=True)
        id_err = norm_dif(xopt, A, b)
        id_avg_time += id_err
        id_time = time.time() - st_time
        id_avg_time += id_time
        if plot:
            print('IDEAL final error: %f (%d iter)' % (id_err, n_iter))
            print('IDEAL time: %f' % id_time)

        st_time = time.time()
        xopt, n_iter, suc, psd_resids = optimize.conjugate_gradient_psd(A, b, full_output=True)
        psd_err = norm_dif(xopt, A, b)
        psd_avg_err += psd_err
        psd_time = time.time() - st_time
        psd_avg_time += psd_time
        if plot:
            print('PSD final error: %f (%d iter)' % (psd_err, n_iter))
            print('PSD time: %f' % psd_time)

        st_time = time.time()
        cgs.A, cgs.b = A, b
        xopt, n_iter, cl_resids = cgs.solve()
        #cl_t, cl_r = [i[1] for i in cl_resids], [i[0] for i in cl_resids]
        cl_err = norm_dif(xopt, A, b)
        cl_avg_err += cl_err
        cl_time = time.time() - st_time
        cl_avg_time += cl_time
        if plot:
            print('Solver final error: %f (%d iter)' % (cl_err, n_iter))
            print('Solver time: %f' % cl_time)

        if plot:
            plt.plot(id_resids.keys(), id_resids.values(), marker='o')
            #plt.plot(psd_resids.keys(), psd_resids.values(), marker='o')
            plt.plot([i[1] for i in cl_resids], [i[0] for i in cl_resids], marker='o')
            plt.yscale('log')
            plt.ylabel('Residual ||Ax - b||')
            plt.xlabel('Time')
            plt.legend(['IDEAL', 'SOLVER'])
            #plt.legend(['IDEAL', 'PSD', 'SOLVER'])
            plt.show()

    psd_avg_time /= 100.0
    psd_avg_err /= 100.0
    id_avg_time /= 100.0
    id_avg_err /= 100.0
    cl_avg_err /= 100.0
    cl_avg_time /= 100.0

    print('IDEAL: ==============')
    print('Avg time: %f' % id_avg_time)
    print('Avg err: %f' % id_avg_err)
    print('PSD: =======')
    print('Avg time: %f' % psd_avg_time)
    print('Avg err: %f' % psd_avg_err)
    print('SOLVER =================')
    print('Avg time: %f' % cl_avg_time)
    print('Avg err: %f' % cl_avg_err)

def visual_IR(start_x=0.0, start_y=0.0):
    A = util.psd_from_cond(cond_num=10000, n=2)
    x_true = 4 * np.random.randn(2)
    b = np.dot(A, x_true)

    start_pos = np.array([float(start_x), float(start_y)])
    print('Initial error: %f' % norm_dif(start_pos, A, b))

    x_opt = optimize.iter_refinement_eps(A, b, x=start_pos, numIter=100)
    path = IR_path(A, b, x=start_pos)
    if la.norm(path[-1] - x_opt) > 0.01:
        print('IR_path and optimize.iter_refine_eps just spit out different minimums.')
        print('IR_path: ' + str(path[-1]))
        print('opt version: ' + str(x_opt))
        sys.exit(0)

    print('Final error: %f' % norm_dif(x_opt, A, b))

    # How wide to view the descent space (Euclidean dist. btwn start and endpoint)
    span = np.sqrt((path[0][0] - x_opt[0])**2 + (path[0][1] - x_opt[1])**2)

    num = 100
    #x1 = x2 = np.linspace(-span, span, num)
    x1 = np.linspace(x_true[0]-span, x_true[0]+span, num)
    x2 = np.linspace(x_true[1]-span, x_true[1]+span, num)
    x1v, x2v = np.meshgrid(x1, x2, indexing='ij', sparse=False)
    hv = np.zeros([num,num])

    for i in range(len(x1)):
        for j in range(len(x2)):
            xx = np.array([x1v[i,j],x2v[i,j]])
            hv[i,j] = np.dot(xx.T,np.dot(A,xx))-np.dot(b.T,xx)
            # f(x) = .5 x.T*A*x - b.T*x

    fig = plt.figure(1)
    ax = fig.gca()
    ll = np.linspace(0.0000000001,4,20)
    ll = 10**ll
    cs = ax.contour(x1v, x2v, hv,levels=ll)
    plt.clabel(cs)
    plt.axis('equal')
    plt.plot([p[0] for p in path], [p[1] for p in path], marker='o', color='pink', markersize=10)
    # RED: true minimum
    plt.plot(x_true[0], x_true[1], marker='o', markersize=18, color='red')
    # GREEN: starting point
    plt.plot(path[0][0], path[0][1], marker='o', markersize=18, color='green')
    plt.legend(['Path', 'Minimum', 'Start'])

    plt.show()

def preconditioned_cg():
    A, b, x_true = util.gen_data(n=1000, cond_num=10**5)
    M = np.diag(np.diag(A))
    print('TRANSFORMED CNUM: %f' % la.cond(np.dot(la.inv(M), A)))

    print('Init resid error: %f\n' % la.norm(b))

    # REGULAR ======================================

    st_time = time.time()
    cg = optimize.ConjugateGradientsSolver(A=A, b=b, full_output=1)
    xopt, n_iter_r, r_resids, r_x_difs = cg.solve(x_true=x_true)
    cg_time = time.time() - st_time
    print('CG final resid err: %f (%d iter)' % (norm_dif(xopt, A, b), n_iter_r))
    print('CG time: %f\n' % cg_time)

    plt.figure(0)
    plt.plot([t for (n,t) in r_resids], [n for (n,t) in r_resids], marker='o')
    plt.figure(1)
    plt.plot([t for (n,t) in r_resids], r_x_difs, marker='o')

    # UNTRANSFORMED PRECONDITIONED ==================
    st_time = time.time()
    M = np.diag(np.diag(A)) # include time to form preconditioner
    un_solver = optimize.AltPreCGSolver(A=A, b=b, M=M, full_output=1)
    xopt, n_iter, u_resids, u_x_difs = un_solver.solve(x_true=x_true)
    un_time = time.time() - st_time
    print('UNTR final resid err: %f (%d iter)' % (norm_dif(xopt, A, b), n_iter))
    print('UNTR time: %f\n' % un_time)

    plt.figure(0)
    plt.plot([t for (n,t) in u_resids], [n for (n,t) in u_resids], marker='o')
    plt.figure(1)
    plt.plot([t for (n,t) in u_resids], u_x_difs, marker='o')

    # ===============================================
    plt.figure(0)
    plt.title('Residuals')
    plt.legend(['Reg', 'Untrans'])
    plt.yscale('log')
    plt.xlabel('Time')
    plt.ylabel('Residual norm')

    plt.figure(1)
    plt.title('Errors')
    plt.legend(['Reg', 'Untrans'])
    plt.yscale('log')
    plt.xlabel('Time')
    plt.ylabel('Error')


    plt.show()

#http://utminers.utep.edu/xzeng/2017spring_math5330/MATH_5330_Computational_Methods_of_Linear_Algebra_files/ln07.pdf
def cgs(n=500, max_iter=500, tol=0.1, cond_num=100):

    A = util.mat_from_cond(cond_num=cond_num, m=n, n=n)
    x_true = 4 * np.random.randn(n)
    b = np.dot(A, x_true)

    init_err = norm_dif(np.zeros(n), A, b)

    #print('Initial resid err: %f' % init_err)


    # START CG SQUARED ============================
    st_time = time.time()
    # Alg 2.2 from above:
    x = np.zeros(n)

    r = b - np.dot(A, x)            # r0 (normal residual)
    cgs_resids = [(la.norm(r), time.time() - st_time)]
    rh = np.copy(r)
    assert np.inner(r, rh) >= 0.0001 # choose rh such that dot(r,rh) != 0
    p, u = np.copy(r), np.copy(r)    # p0, u0

    i = 0
    while i < max_iter:
        i += 1
        a = np.inner(r, rh) / (np.dot(np.dot(A, p), rh)) # Line 4

        q = u - a * np.dot(A, p)            # 5

        x += a * (u + q)                    # 6

        new_r = r - a * np.dot(A, u + q)       # 7
        cgs_resids.append((la.norm(new_r), time.time() - st_time))

        if la.norm(new_r) < tol: break      # 8, 9

        B = np.inner(new_r, rh) / np.inner(r, rh)   # 11

        u = new_r + B * q           # 12

        p = u + B*(q + B*p)

        r = new_r

    cgs_time = time.time() - st_time
    final_err = norm_dif(x, A, b)

    #print('CGS final resid err: %f' % final_err)
    #print('%d iter' % i)
    #print('Took %f sec' % cgs_time)

    # plt.plot([t for (n,t) in cgs_resids], [n for (n,t) in cgs_resids], marker='o')
    # plt.yscale('log')
    # plt.show()

    return final_err < init_err

def system_solve():
    brain = np.load('tomo2D/brain128.npy', 'r')
    f = np.array(brain.flatten(order='F'))

    X = drt.gen_X(n_1=128, n_2=128, m=128**2, sp_rep=1)

    #print('PERCENT NON-ZERO: %f' % (len(np.argwhere(X != 0))  /  float(X.shape[0]*X.shape[1]))  )

    g = X.dot(f)

    # ==============================================================================
    start_time = time.time()
    cgs = optimize.ConjugateGradientsSolver(A=X.T.dot(X), b=X.T.dot(g), full_output=1)
    fopt, n_iter, resids = cgs.solve()
    cg_time = time.time() - start_time
    print('Took %f sec' % cg_time)

    cg_recon = fopt.reshape((128, 128))

    print('final err: %f' % (la.norm(fopt - f) / la.norm(f)))


    plt.plot([t for n,t in resids], [n for n,t in resids], marker='o')
    plt.yscale('log')
    plt.show()



    # ==============================================================================
    plt.figure()
    plt.title('Original')
    plt.imshow(brain)

    plt.figure()
    plt.title('Reconstructed')
    plt.imshow(cg_recon)

    plt.show()

# TODO: Needs major cleaning up.
def x_visual(n=100):
    """
    Plot x estimates from various solvers at each (every 10) iteration(s).
    """

    # Formulate problem ===========================
    A = sps.random(n, n)
    A = A.T.dot(A)      # make positive-definite

    # "box"-shaped true x
    x_true = np.array([50 if (0.4*n)<=i and i<(0.6*n) else 0 for i in range(n)])
    b = A.dot(x_true)
    # =============================================
    # Initialize solvers/get paths ================
    gds = optimize.GradientDescentSolver(A=A, b=b)
    gd_path = gds.path(max_iter=1000)
    cgs = optimize.ConjugateGradientsSolver(A=A, b=b)
    cg_path = cgs.path(max_iter=1000)

    b_x, b_k, b_re, b_path = bfgs.bfgs(A=A, b=b, B=2.0, max_iter=1000)
    # =============================================

    print('num. iterations per solver: ')
    print('GD: %d   CG: %d  BFGS: %d' % (len(gd_path), len(cg_path), len(b_path)))
    print('final residual errors per solver:')
    gd_err, cg_err = norm_dif(gd_path[-1], A, b), norm_dif(cg_path[-1], A, b)
    b_err = norm_dif(b_path[-1], A, b)
    print('GD: %f   CG: %f  BFGS: %f' % (gd_err, cg_err, b_err) )



    for i in range(max(len(gd_path), len(cg_path), len(b_path))):
        if (i%10): continue     # only plot every 10th x
        leg = ['x_true: NA']
        plt.plot(x_true, marker='o', color='blue')

        if i < len(gd_path):
            leg.append('GD: %.1f' % norm_dif(gd_path[i], A, b))
            plt.plot(gd_path[i], marker='o', color='red')

        if i < len(cg_path):
            leg.append('CG: %.1f' % norm_dif(cg_path[i], A, b))
            plt.plot(cg_path[i], marker='o', color='green')

        if i < len(b_path):
            leg.append('BFGS: %.1f' % norm_dif(b_path[i], A, b))
            plt.plot(b_path[i], marker='o', color='purple')

        plt.legend(leg)
        plt.ylim((-10, 60))
        plt.show()

def bfgs_resids(n=100, plot='residuals', sparse=0):
    """
    BFGS residuals and CG residuals.

    Plots residual norms vs time at each iteration.

    Set sparse=True if you want A to be a scipy-sparse matrix.
    """

    # Formulate problem ===========================
    if sparse:
        A = sps.random(n, n)
        A = A.T.dot(A)      # make positive-definite
    else:
        A = util.psd_from_cond(cond_num=10**8, n=n)

    # "box"-shaped true x
    #x_true = np.array([50 if (0.4*n)<=i and i<(0.6*n) else 0 for i in range(n)])
    x_true = np.random.randn(n, )
    b = A.dot(x_true)
    # =============================================
    # Initialize solvers/get resids ===============
    st_time = time.time()
    cgs = optimize.ConjugateGradientsSolver(A=A, b=b, full_output=1)
    cg_x, cg_nit, cg_resid, cg_errs = cgs.solve(max_iter=1000, x_true=x_true)
    cg_time = time.time() - st_time
    print('CG took %f sec' % cg_time)

    st_time = time.time()
    gds = optimize.GradientDescentSolver(A=A, b=b, full_output=1)
    gd_x, gd_nit, gd_resid, gd_errs = gds.solve(max_iter=1000, x_true=x_true)
    gd_time = time.time() - st_time
    print('GD took %f sec' % gd_time)

    st_time = time.time()
    b_x, b_k, b_resid, b_path, b_errs = bfgs.bfgs(A=A, b=b, B=3.0, max_iter=1000, x_true=x_true)
    b_time = time.time() - st_time
    print('BFGS took %f sec' % b_time)

    print('BFGS:CG time ratio: %f\n\n' % (b_time / cg_time))
    # =============================================

    print('num. iterations per solver: ')
    print('CG: %d  GD: %d   BFGS: %d' % (cg_nit, gd_nit, b_k))
    print('final residual errors per solver:')
    cg_err = norm_dif(cg_x, A, b)
    gd_err = norm_dif(gd_x, A, b)
    b_err = norm_dif(b_x, A, b)
    print('CG: %f  GD: %f   BFGS: %f' % (cg_err, gd_err, b_err) )

    plt.xlabel('Time')
    plt.yscale('log')

    if plot == 'residuals':
        plt.ylabel('Residual Norm')
        plt.plot([t for n,t in cg_resid], [n for n,t in cg_resid], marker='o')
        plt.plot([t for n,t in gd_resid], [n for n,t in gd_resid], marker='o')
        plt.plot([t for n,t in b_resid], [n for n,t in b_resid], marker='o')
    elif plot=='errors':
        plt.ylabel('Error Norm')
        plt.plot([t for n,t in cg_resid], cg_errs, marker='o')
        plt.plot([t for n,t in gd_resid], gd_errs, marker='o')
        plt.plot([t for n,t in b_resid], b_errs, marker='o')
    else:
        raise ValueError('allowed values for plot: \'residuals\', \'errors\' ')

    plt.legend(['CG', 'GD', 'BFGS'])
    plt.show()

def bfgs_visual(n=100):
    """
    Plot x estimates from BFGS run w/ various B.
    Use to find optimal B.
    """

    # Formulate problem ===========================
    A = sps.random(n, n)
    A = A.T.dot(A)      # make positive-definite

    # "box"-shaped true x
    x_true = np.array([50 if (0.4*n)<=i and i<(0.6*n) else 0 for i in range(n)])
    b = A.dot(x_true)
    # =============================================
    # Run BFGS w/ various B =======================
    B_range = [float(B) for B in range(1, 11)]
    paths = dict()

    for B in B_range:

        _, _, _, b_path = bfgs.bfgs(A=A, b=b, B=B, max_iter=1000)
        paths[B] = b_path
    # =============================================

    print('num. iterations per B:')
    print([str(B)+': '+str(len(paths[B])) for B in B_range])
    print('final residual errors per B:')
    print([str(B)+': '+str(norm_dif(paths[B][-1], A, b)) for B in B_range])



    #
    # for i in range(max(len(gd_path), len(cg_path), len(b_path))):
    #     if (i%10): continue     # only plot every 10th x
    #     leg = ['x_true: NA']
    #     plt.plot(x_true, marker='o', color='blue')
    #
    #     if i < len(gd_path):
    #         leg.append('GD: %.1f' % norm_dif(gd_path[i], A, b))
    #         plt.plot(gd_path[i], marker='o', color='red')
    #
    #     if i < len(cg_path):
    #         leg.append('CG: %.1f' % norm_dif(cg_path[i], A, b))
    #         plt.plot(cg_path[i], marker='o', color='green')
    #
    #     if i < len(b_path):
    #         leg.append('BFGS: %.1f' % norm_dif(b_path[i], A, b))
    #         plt.plot(b_path[i], marker='o', color='purple')
    #
    #     plt.legend(leg)
    #     plt.ylim((-10, 60))
    #     plt.show()
