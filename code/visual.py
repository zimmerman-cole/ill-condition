import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import util, optimize

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

def visual_gd():
    """
    Visualized gradient descent.
    """
    A = util.psd_from_cond(cond_num=1000,n=2)
    x_true = np.random.randn(2)
    b = np.dot(A,x_true)
    evals,evecs = la.eig(A)
    #print('eigenvalues are: %s' % evals)

    x_opt = optimize.gradient_descent(A,b, x=np.array([2.0,2.0]))
    path = gd_path(A, b, x=np.array([2.0,2.0]))
    #print(path[0])
    span = np.sqrt((path[0][0] - x_opt[0])**2 + (path[0][1] - x_opt[1])**2)
    # print(la.norm(x_true-x_opt))

    num = 100
    # x1 = x2 = np.linspace(-evals[1], evals[0], num)
    x1 = x2 = np.linspace(-span, span, num)
    x1v, x2v = np.meshgrid(x1, x2, indexing='ij', sparse=False)
    hv = np.zeros([num,num])

    for i in range(len(x1)):
        for j in range(len(x2)):
            # hv[i,j] = la.norm(np.dot(A,[x1v[i,j],x2v[i,j]])-b)
            xx = np.array([x1v[i,j],x2v[i,j]])
            hv[i,j] = np.dot(xx.T,np.dot(A,xx))-np.dot(b.T,xx)

    #print(hv)
    fig = plt.figure(1)
    ax = fig.gca()
    # ax.contour(x1v, x2v, hv,50)
    ll = np.linspace(0.0000000001,4,20)
    ll = 10**ll
    cs = ax.contour(x1v, x2v, hv,levels=ll)
    plt.clabel(cs)
    plt.axis('equal')
    plt.plot([p[0] for p in path], [p[1] for p in path], marker='o')
    plt.plot(x_true[0], x_true[1], marker='D', markersize=25) # TRUE POINT
    plt.plot(path[0][0], path[0][1], marker='x', markersize=25) # STARTING POINT
    print('num iter: %d' % len(path))
    plt.show()
