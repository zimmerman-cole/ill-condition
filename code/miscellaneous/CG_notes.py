import numpy as np
import numpy.linalg as la
import matplotlib, colorsys
import matplotlib.pyplot as plt
import scipy.linalg as sla
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.append('..')
import util, pprint
import optimize
from itertools import chain

def coord_descent(A,b,x_0, tol=1e-3, max_iter=10, dirn=0):
    err = np.inf
    iter = 1
    x = np.copy(x_0)
    n = len(x)
    x_true = la.inv(A).dot(b)
    xs = [x_0]
    def f(x):
        return(0.5*x.T.dot(A.dot(x)) - b.T.dot(x))
    def fprime(x):
        g = A.dot(x) - b
        return g

    while(err > tol and iter < max_iter):
        if dirn==0:
            idx = range(n)
        else:
            idx = range(n-1,-1,-1)
        for k in idx:
            gk = np.zeros(n)
            g = b - A.dot(x)
            gk[k] = g[k]
            alpha, _, _, _, _, _ = scipy.optimize.line_search(f, fprime, x, gk)
            x += alpha*gk
            xs.append(np.copy(x))
        err = la.norm(x-x_true)
        iter += 1
    return x, xs

def norm_dif(x, *args):
    """
    Returns || b - Ax ||
    """
    A, b = args
    return la.norm(b - np.dot(A, x))

def gd_path(A, b, x=None):
    """
    Returns points traversed by GD in 2 dimensions.
    """
    assert len(A)==len(A.T)==2
    if x is None:
        x = np.zeros(2)

    pts = [(x[0],x[1])]
    for i in range(100):
        r = b - np.dot(A, x)
        a = np.inner(r.T, r) / float(np.inner(r.T, np.inner(A, r)))
        x += a * r
        pts.append((x[0], x[1]))

        if norm_dif(x, A, b) < 0.00000000000001:
            break

    return pts

def visual_GD(start_x=0.0, start_y=0.0):
    """
    Visualized gradient descent.
    """
    A = util.psd_from_cond(cond_num=1000,n=2)       #
    x_true = 4 * np.random.randn(2)                 # Formulate problem
    b = np.dot(A,x_true)                            #
    start_pos = np.array([float(start_x), float(start_y)])
    print('GD: Initial error: %f' % norm_dif(start_pos, A, b))

    x_opt = optimize.gradient_descent(A,b, x=start_pos)
    path = gd_path(A, b, x=np.array([2.0,2.0]))
    assert la.norm(path[-1] - x_opt) < 0.01 # ensure path tracker gives same minimum
                                            # as "actual" algorithm
    print('GD: Final error: %f (%d iter)' % (norm_dif(x_opt, A, b), len(path)))

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
    plt.plot([p[0] for p in path], [p[1] for p in path], marker='o', color='pink')
    # RED: true minimum
    plt.plot(x_true[0], x_true[1], marker='o', markersize=25, color='red')
    # GREEN: starting point
    plt.plot(path[0][0], path[0][1], marker='o', markersize=25, color='green')
    plt.legend(['Path', 'Minimum', 'Start'])
    plt.show()


def visual_gd_bad_start():
    """
    Visualized gradient descent.
    """
    A = psd_from_cond(cond_num=50,n=2)
    x_true = np.random.randn(2)
    b = np.dot(A,x_true)
    evals,evecs = la.eig(A)

    major_axis = evecs[np.argmax(abs(evals))]
    major_axis[0],major_axis[1] = major_axis[1],major_axis[0]
    minor_axis = evecs[np.argmin(abs(evals))]
    minor_axis[0],minor_axis[1] = minor_axis[1],minor_axis[0]
    worst_axis = max(evals)*major_axis + min(evals)*minor_axis

    y_minor = x_true+minor_axis/la.norm(minor_axis)*5+np.random.randn(2)
    y_major = x_true+major_axis/la.norm(major_axis)*5+np.random.randn(2)*0.05
    y_worst = x_true+worst_axis/la.norm(worst_axis)*5


    print('Minor')
    x_opt_minor = optimize.gradient_descent(A,b, x=np.copy(y_minor))
    path_minor = gd_path(A, b, x=np.copy(y_minor))
    print('Major')
    x_opt_major = optimize.gradient_descent(A,b, x=np.copy(y_major))
    path_major = gd_path(A, b, x=np.copy(y_major))
    print('Worst')
    x_opt_worst = optimize.gradient_descent(A,b, x=np.copy(y_worst))
    path_worst = gd_path(A, b, x=np.copy(y_worst))

    span_minor = np.sqrt((path_minor[0][0] - x_opt_minor[0])**2 + (path_minor[0][1] - x_opt_minor[1])**2)
    span_major = np.sqrt((path_major[0][0] - x_opt_major[0])**2 + (path_major[0][1] - x_opt_major[1])**2)
    span_worst = np.sqrt((path_worst[0][0] - x_opt_worst[0])**2 + (path_worst[0][1] - x_opt_worst[1])**2)
    span = max(span_minor,span_major,span_worst)
    # span = 7

    num = 100
    x1 = x2 = np.linspace(-span, span, num)
    x1v, x2v = np.meshgrid(x1, x2, indexing='ij', sparse=False)
    hv = np.zeros([num,num])

    for i in range(len(x1)):
        for j in range(len(x2)):
            xx = np.array([x1v[i,j],x2v[i,j]])
            hv[i,j] = np.dot(xx.T,np.dot(A,xx))-np.dot(b.T,xx)

    fig = plt.figure(1)
    ax = fig.gca()
    ll = np.linspace(10**-10,4,20)
    ll = 10**ll
    ll = [round(ll[i],0) for i in range(20)]
    cs = ax.contour(x1v, x2v, hv,levels=ll)
    plt.clabel(cs)
    plt.axis('equal')

    # plot true
    plt.plot(x_true[0], x_true[1], marker='D', markersize=10) # TRUE POINT

    # plot paths
    plt.plot([p[0] for p in path_minor], [p[1] for p in path_minor], marker='o', markersize=0.5, color="blue")
    plt.plot(path_minor[0][0], path_minor[0][1], marker='x', markersize=15, color="blue") # STARTING POINT

    plt.plot([p[0] for p in path_major], [p[1] for p in path_major], marker='o', markersize=0.5, color="red")
    plt.plot(path_major[0][0], path_major[0][1], marker='x', markersize=15, color="red") # STARTING POINT

    plt.plot([p[0] for p in path_worst], [p[1] for p in path_worst], marker='o', markersize=0.5, color="green")
    plt.plot(path_worst[0][0], path_worst[0][1], marker='x', markersize=15, color="green") # STARTING POINT

    # plot e-vectors:
    vs_minor = np.array([[ y_minor[0],y_minor[1],major_axis[0],major_axis[1] ] , [ y_minor[0],y_minor[1],minor_axis[0],minor_axis[1] ]])
    vs_major = np.array([[ y_major[0],y_major[1],major_axis[0],major_axis[1] ] , [ y_major[0],y_major[1],minor_axis[0],minor_axis[1] ]])
    vs_worst = np.array([[ y_worst[0],y_worst[1],worst_axis[0],worst_axis[1] ] , [ y_worst[0],y_worst[1],worst_axis[0],worst_axis[1] ]])


    X_minor, Y_minor, U_minor, V_minor = zip(*vs_minor)
    X_major, Y_major, U_major, V_major = zip(*vs_major)
    X_worst, Y_worst, U_worst, V_worst = zip(*vs_worst)

    # plot eigenvectors
    # ax.quiver(X_minor, Y_minor, U_minor, V_minor, angles='xy', scale_units='xy', scale=1, color=["red","blue"])
    # ax.quiver(X_major, Y_major, U_major, V_major, angles='xy', scale_units='xy', scale=1, color=["red","blue"])
    # ax.quiver(X_worst, Y_worst, U_worst, V_worst, angles='xy', scale_units='xy', scale=1, color=["red","blue"])

    plt.draw()
    plt.show()

def visual_GD_CG_no_contour():
    """
    Visual GD and CG    WITHOUT CONTOURS.
    """
    A = util.psd_from_cond(cond_num=1000, n=n)
    x_true = 4 * np.random.randn(n)
    b = np.dot(A, x_true)

    print('Initial resid err: %f' % norm_dif(np.zeros(n), A, b))

    cgs = optimize.ConjugateGradientsSolver(A=A, b=b)
    cg_path = cgs.path()
    print('CG start: ' + str(cg_path[0]))

    gds = optimize.GradientDescentSolver(A=A, b=b)
    gd_path=gds.path()
    print('GD start: ' + str(gd_path[0]))

    plt.plot([x for (x,y) in cg_path], [y for (x,y) in cg_path], marker='o')
    plt.plot([x for (x,y) in gd_path], [y for (x,y) in gd_path], marker='o')
    plt.plot([x_true[0]], [x_true[1]], marker='x', markersize=18)

    plt.legend(['CG', 'GD', 'x_true'])
    plt.title('Paths of GD and CG')
    plt.show()

## ========================================================================== ##
def gen_starts(A,x_true):
    evals,evecs = la.eig(A)
    cond_num = float(la.cond(A))
    print(cond_num)

    ## initialize random starting points
    major_axis = evecs[np.argmax(abs(evals))]
    major_axis[0],major_axis[1] = major_axis[1],major_axis[0]
    minor_axis = evecs[np.argmin(abs(evals))]
    minor_axis[0],minor_axis[1] = minor_axis[1],minor_axis[0]
    worst_axis = max(evals)*major_axis + min(evals)*minor_axis

    ## add noise to minor / major axes (to avoid one-step solutions)
    start_minor = x_true+minor_axis/la.norm(minor_axis)*5+np.random.randn(2)
    start_major = x_true+major_axis/la.norm(major_axis)*5+np.random.randn(2)/cond_num
    start_worst = x_true+worst_axis/la.norm(worst_axis)*5
    # print("start_minor: %s" % start_minor)
    # print("start_major: %s" % start_major)
    # print("start_worst: %s" % start_worst)
    return start_minor, start_major, start_worst

def calc_contours(A,b,spanx1, spanx2):
    ## draw contours of 2*f(x) = 1/2*xAx - bx + c
    num = 100
    x1 = np.linspace(spanx1[0], spanx1[1], num)
    x2 = np.linspace(spanx2[0], spanx2[1], num)
    x1v, x2v = np.meshgrid(x1, x2, indexing='ij', sparse=False)
    hv = np.zeros([num,num])
    for i in range(len(x1)):
        for j in range(len(x2)):
            xx = np.array([x1v[i,j],x2v[i,j]])
            hv[i,j] = np.dot(xx.T,np.dot(A,xx))-2*np.dot(b.T,xx)
    return x1v, x2v, hv

def plot_path(path, path_color):
    plt.plot([p[0] for p in path], [p[1] for p in path], marker='o', markersize=0.5, color=path_color)
    plt.plot(path[0][0], path[0][1], marker='x', markersize=15, color=path_color)

def calc_errs(path,x_true):
    x1_errs = [p[0] - x_true[0] for p in path]
    x2_errs = [p[1] - x_true[1] for p in path]
    errs = zip(x1_errs,x2_errs)
    del errs[-1]
    n_errs = [e/la.norm(e) for e in errs]
    return x1_errs, x2_errs, errs, n_errs

def plot_errs(name, errs, span, scheme=0, space="x"):
    errs_x1, errs_x2 = [e[0] for e in errs], [e[1] for e in errs]

    n = len(errs_x1)
    HSV_tuples = [((x+scheme)*1.0/n, 0.5, 0.5) for x in range(n)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    colors = RGB_tuples
    fig = plt.figure("errs: "+name)
    ax = fig.gca()
    origin = np.array([0,0])
    for i in range(len(errs_x1)):
        plt.quiver(0,0,errs_x1[i],errs_x2[i],color=colors[i], \
                   scale=5,headaxislength=0,headlength=0,label=space+str(i))
    e_span = span*1
    ax.set_ylim([-e_span,e_span])
    ax.set_xlim([-e_span,e_span])
    ip = [np.dot(errs[i],errs[i+1]) for i in range(len(errs)-1)]

    ip = ip[0]
    ax.text(-e_span,e_span-4*scheme,"inner product in "+space+"space: "+str(ip))
    plt.draw()
    plt.legend()

def vis_2x2():
    A = np.asarray([[4., 2.],[1., 1.]])
    b = np.ones(2)
    x_true = la.inv(A).dot(b)
    print(x_true)
    x_0 = np.asarray([1., 5.])

    ## solver object
    s_gd = optimize.GradientDescentSolver(A=A,b=b,full_output=True)
    s_cg = optimize.ConjugateGradientsSolver(A=A,b=b,full_output=True)

    ## paths
    err = [x_0,x_true]
    path_gd = s_gd.path(x_0=x_0)
    path_cg = s_cg.path(x_0=x_0)
    path_diag1 = [x_0, np.array([x_0[0],x_true[1]]), x_true]
    path_diag2 = [x_0, np.array([x_true[0],x_0[1]]), x_true]
    _, coord0 = coord_descent(A,b,x_0, dirn=0)
    _, coord1 = coord_descent(A,b,x_0, dirn=1)
    # path_diag =

    ## errors and residuals
    x_gd, i_gd, resids_gd, errs_gd = s_gd.solve(x_0=x_0, x_true=x_true)
    x_cg, i_cg, resids_cg, errs_cg = s_cg.solve(x_0=x_0, x_true=x_true)

    ## plotting ================================================================
    spanx1 = [-3., 3.]
    spanx2 = [-3., 6.]
    fig = plt.figure("path", figsize=(la.norm(spanx1,1),la.norm(spanx2,1)))

    x1v, x2v, ctrs = CG_notes.calc_contours(A=A,b=b,spanx1=spanx1, spanx2=spanx2)
    ## plot path and contours
    # fig = plt.figure("path")
    ax = fig.gca()
    ll = np.linspace(10**-10,4,20)
    ll = 10**ll
    ll = [round(ll[i],0) for i in range(20)]
    cs = ax.contour(x1v, x2v, ctrs,levels=ll)
    plt.clabel(cs)
    plt.xlabel("x1")
    plt.ylabel("x2")
    # plt.axis('equal')

    plt.scatter(x_0[0], x_0[1], marker='D', s=100, label=r"$x_0$") # START POINT
    plt.scatter(x_true[0], x_true[1], marker='D', s=100, label=r"$x^*$") # TRUE POINT

    # plt.plot([p[0] for p in path_gd], [p[1] for p in path_gd],
    #          marker='o', markersize=0.5, color="blue", label="gd")
    # plt.plot([p[0] for p in path_cg], [p[1] for p in path_cg],
    #          marker='o', markersize=0.5, color="green", label="cg")

    plt.plot([p[0] for p in coord0], [p[1] for p in coord0],
         marker='o', markersize=0.5, color="red", label="coord_desc-x1")
    plt.plot([p[0] for p in coord1], [p[1] for p in coord1],
         marker='o', markersize=0.5, label="coord_desc-x2")

    # plt.plot([p[0] for p in path_diag1], [p[1] for p in path_diag1],
    #      marker='o', markersize=0.5, color="red", label="coord-search-1")
    # plt.plot([p[0] for p in path_diag2], [p[1] for p in path_diag2],
    #      marker='o', markersize=0.5, color="orange", label="coord-search-2")

    plt.plot([p[0] for p in err], [p[1] for p in err],
         marker='o', markersize=0.5, color="orange", label="err")
    plt.legend()
    plt.title("contours of Q")
    plt.grid()
    plt.show()
    fig.savefig("out.pdf")

if __name__ == "__main__":

    # vis_2x2(cond_num=10,solver="ConjugateGradientsSolver",addl_plot=True)
    # vis_2x2(cond_num=30,solver="GradientDescentSolver", addl_plot=False)
    vis_2x2()
