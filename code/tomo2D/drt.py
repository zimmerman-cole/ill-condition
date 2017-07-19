import numpy as np
import pprint

K = 3
M = 10
N = 5

# name  |  type            |  description
# ------------------------------------------------------------------------------
# X     |  [K, M*N] array  |  X-ray transform matrix containing ray lengths
# X_k   |  [1, M*N] array  |  single row of X-ray transform matrix

def siddon_algorithm(X1, Y1, X2, Y2, Xgrid, Ygrid):
    """Uses Siddon's (1985) method to compute cell crossings.
    Source: http://users.elis.ugent.be/~brdsutte/research/publications/1998JCITjacobs.pdf
            https://github.com/aivian/utilities/blob/1390dd0eb5112468bb92c3895b13d4329d0d1553/src/geometry/grid_math.py
    Args:
        X1: the x coordinate of the first point
        Y1: the y coordinate of the first point
        X2: the x coordinate of the end point
        Y2: the y coordinate of the end point
        Xgrid: the x coordinates of the gridlines dividing cells
        Ygrid: the y coordinate of the gridlines dividing cells
    Returns:
        A list of two numpy arrays
        cells: the first array is the i coordinate
            and the second column is the j coordinate
            of the cell corresponding to l.
        l: the length of the full line that lies in each cell
        (cells, l)
        Note that if a length of the path falls outside of the grid, cells will
        return [-1,-1] in the appropriate row
    """
    # make Xgrid and Ygrid numpy arrays (harmless if it's already a numpy array)
    Xgrid = np.array(Xgrid)
    Ygrid = np.array(Ygrid)
    # specify the distance between planes (should be regular here)
    dx = np.abs(Xgrid[1] - Xgrid[0])
    dy = np.abs(Ygrid[1] - Ygrid[0])
    # find the number of grid lines dividing cells, note that there are (Nx-1,Ny-1) voxels in this 2d array
    Nx = Xgrid.size
    Ny = Ygrid.size

    print("Nx, Ny: %s, %s" % (Nx,Ny))

    # calculate the range of parametric values
    if (X2-X1) != 0.0:
        ax = np.zeros(Nx)
        ax[0] = (Xgrid[0]-X1)/(X2-X1)
        print("Xgrid[0]: %s" % Xgrid[0])
        ax[Nx-1] = (Xgrid[Nx-1]-X1)/(X2-X1)
    else:
        ax = np.zeros(0)
    if (Y2-Y1) != 0.0:
        ay = np.zeros(Ny)
        ay[0] = (Ygrid[0]-Y1)/(Y2-Y1)
        print("Ygrid[0]: %s" % Ygrid[0])
        ay[Ny-1] = (Ygrid[Ny-1]-Y1)/(Y2-Y1)
    else:
        ay = np.zeros(0)

    print("ax, ay: %s, %s" % (ax,ay))
    print("len ax, ay: %s, %s" % (len(ax),len(ay)))

    if (ax.size > 0) and (ay.size > 0):
        amin = max([0.0, min(ax[0], ax[Nx-1]), min(ay[0], ay[Ny-1])])
        amax = min([1.0, max(ax[0], ax[Nx-1]), max(ay[0], ay[Ny-1])])
    elif (ax.size == 0) and (ay.size > 0):
        amin = max([0, min(ay[0], ay[Ny-1])])
        amax = min([1, max(ay[0], ay[Ny-1])])
    elif (ay.size == 0) and (ax.size > 0):
        amin = max([0, min(ax[0], ax[Nx-1])])
        amax = min([1, max(ax[0], ax[Nx-1])])
    else:
        amin = 0.0
        amax = 1.0

    print("amin, amax: %s, %s" % (amin,amax))

    # Calculate the range of indices covered
    if (X2-X1)>=0:
        imin = Nx - np.floor((Xgrid[Nx-1] - amin*(X2-X1) - X1)/dx)
        imax = 1 + np.floor((X1 + amax*(X2-X1) - Xgrid[0])/dx)
    else:
        imin = Nx - np.floor((Xgrid[Nx-1] - amax*(X2-X1) - X1)/dx)
        imax = 1 + np.floor((X1 + amin*(X2-X1) - Xgrid[0])/dx)
    if (Y2-Y1)>=0:
        jmin = Ny - np.floor((Ygrid[Ny-1] - amin*(Y2-Y1) - Y1)/dy)
        jmax = 1 + np.floor((Y1 + amax*(Y2-Y1) - Ygrid[0])/dy)
    else:
        jmin = Ny - np.floor((Ygrid[Ny-1] - amax*(Y2-Y1) - Y1)/dy)
        jmax = 1 + np.floor((Y1 + amin*(Y2-Y1) - Ygrid[0])/dy)


    print("imin, imax: %s, %s" % (imin,imax))

    # Calculate parametric sets
    if ax.size > 0:
        i = int(imin)
        for p in range(0, (int(imax-imin)+1)):
            # print("imin = %s" % imin)
            ax[p] = (Xgrid[i-1]-X1)/(X2-X1)
            i = i + 1
        ax = ax[0:(int(imax-imin)+1)]
    if ay.size > 0:
        j = int(jmin)
        for p in range(0, (int(jmax-jmin)+1)):
            # print("jmin = %s" % jmin)
            ay[p] = (Ygrid[j-1]-Y1)/(Y2-Y1)
            j = j + 1
        ay = ay[0:(int(jmax-jmin)+1)]

    # merge sets to form a
    alpha = np.unique(np.hstack([amin, ax, ay, amax]))

    # distance from point 1 to point 2
    d12 = np.sqrt((X2-X1)**2.0+(Y2-Y1)**2.0)

    # calculate voxel lengths
    # The pixel that contains the midpoint of the intersections that bound a
    # length contains the entirety of that length. We use this obvious fact
    # to return the indices of cells crossed by the vector
    l = np.zeros(alpha.size).astype(float)
    i = np.zeros(alpha.size).astype(int)
    j = np.zeros(alpha.size).astype(int)
    for m in range(1, alpha.size):
        l[m] = d12*(alpha[m]-alpha[m-1]);
        # find the midpoint of each length
        amid = (alpha[m]+alpha[m-1])/2.0;
        # Find the x index
        i[m] = np.floor((X1 + amid*(X2-X1)-Xgrid[0])/dx) # 0 indexed, otherwise +1
        # find the y index
        j[m] = np.floor((Y1 + amid*(Y2-Y1)-Ygrid[0])/dy) # 0 indexed, otherwise +1
    # remove the first index
    l = np.delete(l, 0)
    i = np.delete(i, 0)
    j = np.delete(j, 0)

    # now lets deal with the case when the end point is outside of the grid
    if amax < 1.0:
        arem = 1-amax
        l = np.append(l, (arem*d12))
        i = np.append(i, -1)
        j = np.append(j, -1)

    # and of course the case where the start point is outside of the grid
    if amin > 0.0:
        arem = amin
        l = np.insert(l, 0, (arem*d12))
        i = np.insert(i, 0, -1)
        j = np.insert(j, 0, -1)

    # put cells together as a tuple to make indexing obvious
    cells = (i, j)

    # put lengths in cell matrix
    inds = zip(i,j)
    print("inds: %s" % inds)
    A_k = np.zeros([Nx,Ny])
    print("A_k shape", A_k.shape)
    for ind_num in range(len(inds)):
        i = inds[ind_num][0]
        j = inds[ind_num][1]
        A_k[i,j] = l[ind_num]
    for row in range(Nx):
        # pprint.pprint(A_k[row,:])
        print([round(A_k[row,i],2) for i in range(Ny)])
    # return the cells and the lengths
    return cells, l

K = 2
M = 8
N = 4

thetas = np.linspace(0, np.pi, K+1)[0:K]
ms = np.tan(thetas)
print("thetas: %s" % thetas)
print("slopes: %s" % ms)

def gen_grids(M, N):
    """
    generates Xgrid and Ygrid from even M, N
    """
    Xgrid = 1.0*np.arange(-int(N/2), int(N/2)+1)
    Ygrid = 1.0*np.arange(-int(M/2), int(M/2)+1)
    return Xgrid, Ygrid

Xgrid, Ygrid = gen_grids(M,N)
print(Xgrid)
print(Ygrid)

def gen_eep(Xgrid, Ygrid, thetas):
    """
    purpose:
    ----------
    generates entry/exit points (eep) for a given X-Y grid
    for every `theta` in `thetas`

    params:
    ----------
    Xgrid: np.array containing discretized x values
    Ygrid: np.array containing discretized y values
    thetas: list of angles in [0,2pi]

    output
    ----------
    eepoints: list containing tuples of (x_min, y_min, x_max, y_max)
    """

    x_max = Xgrid[-1]
    y_max = Ygrid[-1]
    s = y_max/x_max

    points = []
    for theta in thetas:
        m = float(np.tan(theta))
        if np.absolute(m) <= s:
            x_max = float(Xgrid[-1])
            x_min = float(Xgrid[0])
            y_max = float(m * x_max)
            y_min = float(m * x_min)
            pp = [x_min, y_min, x_max, y_max]
            pp = [round(p,5) for p in pp]
            points.append(pp)
        else:
            x_max = float(Ygrid[-1]/m)
            x_min = float(Ygrid[0]/m)
            y_max = float(Ygrid[-1])
            y_min = float(Ygrid[0])
            pp = [x_min, y_min, x_max, y_max]
            pp = [round(p,5) for p in pp]
            points.append(pp)
    return points

points = gen_eep(Xgrid=Xgrid, Ygrid=Ygrid, thetas=thetas)
print(points)

x1,y1,x2,y2 = points[1]
x1,y1,x2,y2 = -2.,-3.,2.,3.
cells, l = siddon_algorithm(x1,y1,x2,y2,Xgrid,Ygrid)
print(points[1])
print(cells)
print(l)

def gen_X(M=None, N=None, K=None, thetas=None):
    # TODO: FINISH
    if M % 2 == 1:
        M += 1
    if N % 2 == 1:
        N += 1
    M = int(M)
    N = int(N)
    K = int(K)
    thetas = np.linspace(0, np.pi, K+1)[0:K]
    Xgrid, Ygrid = gen_grids(M,N)
    for theta in thetas:
        pass
