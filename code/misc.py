import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import time, optimize, util

"""
Temporary holding place for stuff /// stuff that doesn't fit anywhere else.
"""

# || b - Ax ||
def norm_dif(x, *args):
    """
    Return || b - Ax || (Frobenius norm).
    """
    A, b = args
    return la.norm(b - np.dot(A, x))

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
        (Continuation)
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
