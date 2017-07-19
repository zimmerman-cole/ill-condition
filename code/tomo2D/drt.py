import numpy as np
import scipy.sparse as sparse
import pprint


# name        |  type                  |  description
# ------------------------------------------------------------------------------
# X           |  [m, n_1 * n_2] array  |  X-ray transform matrix of ray line ints
# X_k         |  [1, n_1 * n_2] array  |  single row of X-ray transform matrix
# n_1         |  int (even)            |  rows of image
# n_2         |  int (even)            |  cols of image
# m           |  int                   |  number of X-rays fired; divides (0,pi)
# x_grid      |  [n_1+1, 1] array      |  n_1+1 gridlines for n_1 pixels
# y_grid      |  [n_2+1, 1] array      |  n_2+1 gridlines for n_2 pixels
# theta       |  [m, 1] array          |  angles dividin (0,pi)
# i: 1...n_1  |  int                   |  counting index for n_1 (rows of image)
# j: 1...n_2  |  int                   |  counting index for n_2 (cols of image)
# k: 1...m    |  int                   |  counting index for m (rays of X-ray)

def siddon_algorithm(x1, y1, x2, y2, x_grid, y_grid, debug=False):
    """Uses Siddon's (1985) method to compute cell crossings.
    Source: http://users.elis.ugent.be/~brdsutte/research/publications/1998JCITjacobs.pdf
            https://github.com/aivian/utilities/blob/1390dd0eb5112468bb92c3895b13d4329d0d1553/src/geometry/grid_math.py
    Args:
        x1: the x coordinate of the first point
        y1: the y coordinate of the first point
        x2: the x coordinate of the end point
        y2: the y coordinate of the end point
        x_grid: the x coordinates of the gridlines dividing cells
        y_grid: the y coordinate of the gridlines dividing cells
    Returns:
        X_k: k-th 1 x n_1*n_2 array flattened by row
        Note that if a length of the path falls outside of the grid, cells will
        return [-1,-1] in the appropriate row
    """
    # make x_grid and y_grid numpy arrays (harmless if it's already a numpy array)
    x_grid = np.array(x_grid)
    y_grid = np.array(y_grid)
    # specify the distance between planes (should be regular here)
    dx = np.abs(x_grid[1] - x_grid[0])
    dy = np.abs(y_grid[1] - y_grid[0])
    # find the number of grid lines dividing cells, note that there are (Nx-1,Ny-1) voxels in this 2d array
    Nx = x_grid.size
    Ny = y_grid.size
    if debug:
        print("Nx, Ny: %s, %s" % (Nx,Ny))

    # calculate the range of parametric values
    if (x2-x1) != 0.0:
        ax = np.zeros(Nx)
        ax[0] = (x_grid[0]-x1)/(x2-x1)
        if debug:
            print("x_grid[0]: %s" % x_grid[0])
        ax[Nx-1] = (x_grid[Nx-1]-x1)/(x2-x1)
    else:
        ax = np.zeros(0)
    if (y2-y1) != 0.0:
        ay = np.zeros(Ny)
        ay[0] = (y_grid[0]-y1)/(y2-y1)
        if debug:
            print("y_grid[0]: %s" % y_grid[0])
        ay[Ny-1] = (y_grid[Ny-1]-y1)/(y2-y1)
    else:
        ay = np.zeros(0)

    if debug:
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

    if debug:
        print("amin, amax: %s, %s" % (amin,amax))

    # Calculate the range of indices covered
    if (x2-x1)>=0:
        imin = Nx - np.floor((x_grid[Nx-1] - amin*(x2-x1) - x1)/dx)
        imax = 1 + np.floor((x1 + amax*(x2-x1) - x_grid[0])/dx)
    else:
        imin = Nx - np.floor((x_grid[Nx-1] - amax*(x2-x1) - x1)/dx)
        imax = 1 + np.floor((x1 + amin*(x2-x1) - x_grid[0])/dx)
    if (y2-y1)>=0:
        jmin = Ny - np.floor((y_grid[Ny-1] - amin*(y2-y1) - y1)/dy)
        jmax = 1 + np.floor((y1 + amax*(y2-y1) - y_grid[0])/dy)
    else:
        jmin = Ny - np.floor((y_grid[Ny-1] - amax*(y2-y1) - y1)/dy)
        jmax = 1 + np.floor((y1 + amin*(y2-y1) - y_grid[0])/dy)


    if debug:
        print("imin, imax: %s, %s" % (imin,imax))

    # Calculate parametric sets
    if ax.size > 0:
        i = int(imin)
        for p in range(0, (int(imax-imin)+1)):
            # print("imin = %s" % imin)
            ax[p] = (x_grid[i-1]-x1)/(x2-x1)
            i = i + 1
        ax = ax[0:(int(imax-imin)+1)]
    if ay.size > 0:
        j = int(jmin)
        for p in range(0, (int(jmax-jmin)+1)):
            # print("jmin = %s" % jmin)
            ay[p] = (y_grid[j-1]-y1)/(y2-y1)
            j = j + 1
        ay = ay[0:(int(jmax-jmin)+1)]

    # merge sets to form a
    alpha = np.unique(np.hstack([amin, ax, ay, amax]))

    # distance from point 1 to point 2
    d12 = np.sqrt((x2-x1)**2.0+(y2-y1)**2.0)

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
        i[m] = np.floor((x1 + amid*(x2-x1)-x_grid[0])/dx) # 0 indexed, otherwise +1
        # find the y index
        j[m] = np.floor((y1 + amid*(y2-y1)-y_grid[0])/dy) # 0 indexed, otherwise +1
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
    X_k = np.zeros([Nx-1,Ny-1])

    for ind_num in range(len(inds)):
        i = inds[ind_num][0]
        j = inds[ind_num][1]
        X_k[i,j] = l[ind_num]

    # printing
    if debug:
        print("inds: %s" % inds)
        print("X_k shape", X_k.shape)
        for row in range(Nx-1):
            # pprint.pprint(X_k[row,:])
            print([round(X_k[row,i],2) for i in range(Ny-1)])

    # return X_k
    X_k = X_k.flatten()
    return X_k

def gen_grids(n_1, n_2):
    """
    generates x_grid and y_grid from even n_1, n_2
    """
    x_grid = 1.0*np.arange(-int(n_2/2), int(n_2/2)+1)
    y_grid = 1.0*np.arange(-int(n_1/2), int(n_1/2)+1)
    return x_grid, y_grid

def gen_eep(x_grid, y_grid, thetas):
    """
    purpose:
    ----------
    generates entry/exit points (eep) for a given X-Y grid
    for every `theta` in `thetas`

    params:
    ----------
    x_grid: np.array containing discretized x values
    y_grid: np.array containing discretized y values
    thetas: list of angles in [0,2pi]

    output
    ----------
    eepoints: list containing tuples of (x_min, y_min, x_max, y_max)
    """

    x_max = x_grid[-1]
    y_max = y_grid[-1]
    s = y_max/x_max

    points = []
    for theta in thetas:
        m = float(np.tan(theta))
        if np.absolute(m) <= s:
            x_max = float(x_grid[-1])
            x_min = float(x_grid[0])
            y_max = float(m * x_max)
            y_min = float(m * x_min)
            pp = [x_min, y_min, x_max, y_max]
            pp = [round(p,5) for p in pp]
            points.append(pp)
        else:
            x_max = float(y_grid[-1]/m)
            x_min = float(y_grid[0]/m)
            y_max = float(y_grid[-1])
            y_min = float(y_grid[0])
            pp = [x_min, y_min, x_max, y_max]
            pp = [round(p,5) for p in pp]
            points.append(pp)
    return points

def gen_X(n_1=None, n_2=None, m=None, sp_rep=False, debug=False):
    ## handle args
    if n_1 % 2 == 1:
        n_1 += 1
    if n_2 % 2 == 1:
        n_2 += 1
    n_1 = int(n_1)
    n_2 = int(n_2)
    m = int(m)

    ## generate grid
    xgrid, ygrid = gen_grids(n_1,n_2)

    ## partition [0,pi] with m angles
    theta = np.linspace(0, np.pi, m+1)[0:m]

    ## compute entry/exit points on grid for each slope
    eepoints = gen_eep(x_grid=xgrid, y_grid=ygrid, thetas=theta)

    ## initialize X-ray transform matrix
    X = np.zeros([m, n_1*n_2])
    for k in range(m):
        x1,y1, x2,y2 = eepoints[k]
        X[k,:] = siddon_algorithm(x1, y1, x2, y2, xgrid, ygrid, debug=debug)
    if sp_rep:
        return(sparse.csr_matrix(X))
    else:
        return(X)

X = gen_X(n_1=8, n_2=4, m=3, sp_rep=False, debug=False)
print(X)
