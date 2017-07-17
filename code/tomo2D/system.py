import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def vis(px, c_center, c_rad, rays=[]):
    """
    Shows circle from which rays are fired, image pixels contained within,
        ray origin points.

    px: array of Pixel objects.
    c_center: tuple containing coordinates of circle center.
    c_rad: radius of circle
    """
    fig, ax = plt.subplots()

    # x- and y-coords of top-left corners of each pixel
    xs = [p.x for p in px]
    ys = [p.y for p in px]

    ax.plot(xs, ys, marker='o')

    circle = plt.Circle(c_center, c_rad, color='r')

    ax.add_artist(circle)
    ax.axis([c_center[0] - 1.25*c_rad, c_center[0] + 1.25 * c_rad,
                c_center[1] - 1.25*c_rad, c_center[1] + 1.25 * c_rad])

    ar_size = 10.0   # arrow size ~~ (1 / ar_size)
    for r in rays:
        ax.arrow(x=r.ox, y=r.oy, dx=-r.dx/ar_size, dy=-r.dy/ar_size)
        #ax.plot(r.ox, r.oy, marker='x')

    plt.show()

class Const:
    """
    Constants
    """
    # Center of circle around which rays are fired
    cx, cy = None, None
    # Radius of circle
    rad = None

    # Pixel size
    ps = None

class Ray:

    def __init__(self, angle):
        # angle measured from positive end of x-axis
        assert 0 <= angle and angle <= 2*np.pi

        # distance from circle center to ray "origin"
        # (RELATIVE COORDINATES)
        self.dx = np.cos(angle)*Const.rad
        self.dy = np.sin(angle)*Const.rad
        # print(self.dx, self.dy)
        # raw_input()

        # ABSOLUTE coordinates of point from which ray is fired ("origin")
        self.ox = Const.cx + self.dx
        self.oy = Const.cy + self.dy

        self.angle = angle

    def _intersects_x(self, x):
        pass

    def _intersects_y(self, y):
        pass

class Pixel:

    def __init__(self, x, y):
        """
        (x,y): ABSOLUTE coordinates of TOP-LEFT corner of pixel.
                ( (0,0) is bottom-left corner of bottom-left-most pixel) )
        """
        self.x, self.y = x, y

        # RELATIVE coordinates of TOP-LEFT of pixel (relative to center of circle)
        self.dx = (self.x - Const.cx) if self.x > 0 else (self.x + Const.cx)
        self.dy = (self.y - Const.cy) if self.y > 0 else (self.y + Const.cy)


    def _intersect(self, ray):
        pass


class Image:
    """
                                Each row of X tracks one ray's intersection
    Phantom layout:                 with each of these n^2 pixels:
    (0,0) (0,1) ... (0,n)       (0,0)    (0,1)   ...  (0, n^2)
    (1,0) (1,1) ... (1,n)       (1,0)    (1,1)   ...  (1, n^2)
      ...  ...  ...  ...          ...     ...    ...    ...
    (n,0) (n,1) ... (n,n)       (n^2,0) (n^2,1)  ...  (n^2, n^2)

    ^ (f, but flattened)

    """

    def __init__(self, n, ps=100):
        """
        ps: pixel size
        """
        #
        self.x_ranges = RangeDict()
        for i in range(0, n-1):
            pass
        # corresponding y-ranges


# load phantom. MUST BE SQUARE
filename = 'brain128.npy'
phantom = np.load(filename, 'r')
assert len(phantom) == len(phantom.T)

# NOTE: coordinate system used in calculating system matrix is
#       standard (x,y) axes, where each pixel is a square of size: (ps, ps),
#       and the origin (0,0) is the bottom-left corner of the bottom-left-most pixel.
#       Rays are fired in a circle around the image (a square arrangement of pixels)
#       on a circle whose center is the center of the image.
#
#       For now at least, only one ray is fired from each position around the circle
#           (no cone beam, no parallel beam).
#       Pixel aspect ratio is fixed at (1:1)

img_len = len(phantom)
Const.ps = 100.0 # pixel size
_max_ = img_len * Const.ps
print('coord of extreme top-right of image: (%f, %f)' % (_max_, _max_))

# circle-center coordinates
Const.cx, Const.cy = Const.ps*img_len / 2.0, Const.ps*img_len / 2.0
print('circle center: (%f, %f)' % (Const.cx, Const.cy))
# circle radius (1.65 * dist(circle center, corner of image-square))
Const.rad = 1.65 * abs(_max_ - Const.cx)
print('circle radius: %f' % Const.rad)

# =======================================================
# coords: (x,y) coordinates of TOP-LEFT (!!!) corner of every pixel.
# Origin is bottom-left corner of bottom-left-most pixel.

# EXAMPLE: for ps=100 and img_len=128:
# (0, 12800) (100, 12800) ... (12800, 12800)
#   ...                   ...
# (0,   100) (100,   100) ... (12800,   100)
# (0,     0) (0,     100) ... (12800,   100)
coords = [ [(i, j) for i in range(0, int(_max_), int(Const.ps))] for j in range(int(_max_), 0, int(-Const.ps))]
# TODO: ^ this is a mess
# ^ coords of top-left corner of each pixel
# =======================================================

print('\nTOP-LEFT COORDS OF BOTTOM-LEFT-MOST 5x5:')
for row in coords[-5:]:
    print(row[:5])

# create array of PIXEL objects (class defined above)
px = []
for row in coords:
    for x, y in row:
        px.append(Pixel(x, y))

rays = []
n_rays = 8
ray_st_s = (2.0 * np.pi) / n_rays # ray "step size"
for ang in [ray_st_s*j for j in range(n_rays)]:
    rays.append(Ray(ang))

print('\nRay origins:')
for r in rays:
    print(r.ox, r.oy)

vis(px, (Const.cx, Const.cy), Const.rad, rays=rays)






















pass
