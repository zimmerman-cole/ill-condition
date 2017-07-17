import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import sys

""" NOTE:

In this file, there are two "coordinate systems" used. Both use (x,y), but
differ in what their origins are.
   o "ABSOLUTE" coordinates: origin is bottom-left corner of bottom-left-most
            pixel in image
   o "RELATIVE" coordinates: origin is center of circle from which rays are
            fired (this is in dead center of image array). Used

=== TERMS USED IN COMMENTS ===
-CC:    x and/or y coordinate(s) of circle center

"""

def vis(px, c_center, c_rad, rays=[], \
            pts=None):
    """
    Shows circle from which rays are fired, image pixels contained within,
        ray origin points.

    px: array of Pixel objects.
    c_center: tuple containing coordinates of circle center.
    c_rad: radius of circle.

    rays: array of rays (plot their origin firing-points)
    intersect_pts: additional points to plot ( used to plot points of
                    intersection between ray(s) and pixel(s) )
    """
    fig, ax = plt.subplots()

    # x- and y-coords of top-left corners of each pixel
    px_flat = np.array(px).flatten()
    xs = [p.x for p in px_flat]
    ys = [p.y for p in px_flat]

    ax.plot(xs, ys, marker='o')

    circle = plt.Circle(c_center, c_rad, color='r')

    ax.add_artist(circle)
    ax.axis([c_center[0] - 1.25*c_rad, c_center[0] + 1.25 * c_rad,
                c_center[1] - 1.25*c_rad, c_center[1] + 1.25 * c_rad])

    ar_size = 10.0   # arrow size ~~ (1 / ar_size)
    for r in rays:
        ax.arrow(x=r.ox, y=r.oy, dx=-r.dx/ar_size, dy=-r.dy/ar_size)
        #ax.plot(r.ox, r.oy, marker='x')

    if (rays != []) and (pts is not None):
        ax.plot([x for x, y in pts], [y for x, y in pts], marker='o')

    plt.show()

class Const:
    """
    Container for constants.
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

        # distance from CC to ray "origin"
        # (RELATIVE COORDINATES)
        self.dx = np.cos(angle)*Const.rad
        self.dy = np.sin(angle)*Const.rad
        # print(self.dx, self.dy)
        # raw_input()

        # ABSOLUTE coordinates of point from which ray is fired ("origin")
        self.ox = Const.cx + self.dx
        self.oy = Const.cy + self.dy

        self.angle = angle

        print('ang: %f' % self.angle)

        # Check if this ray shoots vertically, horizontally, or at an angle.
        if abs(angle) < 0.00001 or abs(angle - np.pi) < 0.00001:
            # horizontal
            self.type = 'H'
        elif abs(angle - np.pi/2.0) < 0.00001 or abs(angle - (3*np.pi)/2.0) < 0.00001:
            # vertical
            self.type = 'V'
        else:
            # at an angle
            self.type = 'A'

        print('type: ' + self.type)
        raw_input()

    def _intersects_x(self, ddx):
        """
        Calculate ray's y-value at given x=cx+ddx.
            (cx: x-coordinate of CC)
        """

        # if this ray is VERTICAL
        if self.type == 'V':
            return 'V'

        return float(ddx) * np.tan(self.angle)

    def _intersects_y(self, ddy):
        """
        Calculate ray's x-value at given y=cy+ddy.
            (cy: y-coordinate of CC)
        """
        # if this ray is HORIZONTAL
        if self.type == 'H':
            return 'H'

        return ddy / (1.0 / np.tan(self.angle))

class Pixel:

    def __init__(self, x, y, idx):
        """
        (x,y): ABSOLUTE coordinates of TOP-LEFT corner of pixel.
                ( (0,0) is bottom-left corner of bottom-left-most pixel) )

        idx:    tuple containing pixel's position in the image array
                    i.e. (row_num, col_num)
        """
        self.x, self.y = x, y

        self.idx = idx

        # RELATIVE coordinates of TOP-LEFT of pixel (relative to center of circle)
        self.dx = (self.x - Const.cx) if self.x > 0 else (self.x + Const.cx)
        self.dy = (self.y - Const.cy) if self.y > 0 else (self.y + Const.cy)


    def _intersect(self, ray):
        pass

class ImageArray:

    def __init__(self, px):
        self.px = px
        rows = {}



# NOTE: coordinate system used in calculating system matrix is
#       standard (x,y) axes, where each pixel is a square of size: (ps, ps),
#       and the origin (0,0) is the bottom-left corner of the bottom-left-most pixel.
#       Rays are fired in a circle around the image (a square arrangement of pixels)
#       on a circle whose center is the center of the image.
#
#       For now at least, only one ray is fired from each position around the circle
#           (no cone beam, no parallel beam).
#       Pixel aspect ratio is fixed at (1:1)

img_len = 5
Const.ps = 100.0 # pixel size
_max_ = img_len * Const.ps
print('coord of extreme top-right of image: (%f, %f)' % (_max_, _max_))

# circle-center coordinates (referred to as CC throughout comments)
Const.cx, Const.cy = Const.ps*img_len / 2.0, Const.ps*img_len / 2.0
print('circle center: (%f, %f)' % (Const.cx, Const.cy))
# circle radius (1.65 * dist(CC, corner of image-square))
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

# create 2D array of PIXEL objects (class defined above)
px = []
for i in range(len(coords)):
    px.append([])
    j = 0
    for x, y in coords[i]:
        px[i].append( Pixel( x, y, (i,j) ) )
        j += 1

rays = []
n_rays = 18
arc_range = 1.0 * np.pi

ray_st_s = arc_range / float(n_rays) # ray "step size"
for ang in [ray_st_s*j for j in range(n_rays)]:
    # print('ang: %f' % ang)
    rays.append(Ray(ang))

    # raw_input()

# print('\nRay origins:')
# for r in rays:
#     print(r.ox, r.oy)

# intersections for each ray
ray_Xs = []
ray_num = 0

# for each ray:
for r in rays:

    # for convenience, assign an (img_len x img_len) array to each ray
    # (will flatten down later to actual X)
    ray_Xs.append(np.zeros((img_len, img_len)))

    if r.type == 'H':
        # for HORIZONTAL rays, just grab the row of pixels whose
        #   ybottom <= ray.y < ytop. For those pixels, the length of
        #   intersection is equal to the length of the pixel. For all other
        #   pixels, the length of intersection is 0.

        for row_num in range(len(px)):    # locate the row of pixels
            if (px[row_num][0].dy - 100) <= r.dy < px[row_num][0].dy:

                # mark down the row's intersection lengths
                for i in range(len(px[row_num])):
                    ray_Xs[ray_num][row_num][i] = Const.ps



    elif r.type == 'V':
        # same idea for VERTICAL rays
        for col_num in range(len(px)):    # locate the row of pixels
            if (px[0][col_num].dx - 100) <= r.dx < px[0][col_num].dx:
                # mark down the row's intersection lengths
                for i in range(len(px[col_num])):
                    ray_Xs[ray_num][i][col_num] = Const.ps
    else:
        # for NON-HORIZONTAL and NON-VERTICAL rays

        # for each row of pixels:
        # for p_row in px:
        #     ddy = p_row[0].dy   # y-coord of this row's top side (w/ respect to CC)
        #
        #
        #     x_intersec = r._intersects_y(ddy)  # first, find x-value where this ray
        #                                         # intersects this row's top side
        #
        #     vis(px, (Const.cx, Const.cy), Const.rad, rays=[rays[0]], pts=[(Const.cx+x_intersec, Const.cy+ddy)])
        #     sys.exit(0)
        pass

    ray_num += 1

for ray_num in range(len(ray_Xs)):
    print('ray_num: %d  ang: %f ' % (ray_num, rays[ray_num].angle  ))
    print('type: ' + rays[ray_num].type)
    print(ray_Xs[ray_num])
    raw_input()


# for ray_n in range(len(ray_Xs)):
#     ray_Xs[ray_n] = ray_Xs[ray_n].flatten()
#
# ray_Xs = np.array(ray_Xs)

#print(ray_Xs)























pass
