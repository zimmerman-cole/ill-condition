import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import util, optimize, sys

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
    A = util.psd_from_cond(cond_num=50,n=2)
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
    

    x_opt_minor = optimize.gradient_descent(A,b, x=np.copy(y_minor))
    path_minor = gd_path(A, b, x=np.copy(y_minor))
    x_opt_major = optimize.gradient_descent(A,b, x=np.copy(y_major))
    path_major = gd_path(A, b, x=np.copy(y_major))
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
    ax.quiver(X_minor, Y_minor, U_minor, V_minor, angles='xy', scale_units='xy', scale=1, color=["red","blue"])
    ax.quiver(X_major, Y_major, U_major, V_major, angles='xy', scale_units='xy', scale=1, color=["red","blue"])
    ax.quiver(X_worst, Y_worst, U_worst, V_worst, angles='xy', scale_units='xy', scale=1, color=["red","blue"])
    plt.draw()
    plt.show()

