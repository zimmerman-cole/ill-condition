import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

# class RangeDict(dict):
#
#     def __init__(self, d={}):
#         for k, v in d.items():
#             self[k] = v
#
#     def __getitem__(self, key):
#         """
#         Key must be a SINGLE integer/float.
#         """
#         for k, v in self.items():
#             if k[0] <= key < k[1]:
#                 return v
#         raise KeyError('Key %s not in any key-range.' % key)
#
#     def __setitem__(self, key, value):
#         """
#         Key must be an ITERABLE of LENGTH TWO.
#         """
#         try:
#             if len(key) == 2:
#                 if key[0] < key[1]:
#                     dict.__setitem__(self, key, value)
#                 else:
#                     raise ValueError('First element of key must be less than second element.')
#             else:
#                 raise ValueError('Key must be iterable of length two.')
#         except TypeError:
#             raise TypeError('Key must be iterable of length two.')
#
# class Pixel:
#
#     def __init__(self, left_x, top_y):
#         # left_x: pixel index
#         self.left_x = int(left_x)
#         self.top_y = int(top_y)

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
        self.dy = np.cos(angle)*Const.rad

        # ABSOLUTE coordinates of point from which ray is fired ("origin")
        self.ox = Const.cx + dx
        self.oy = Const.cy + dy

        self.angle = angle

    def _intersects_x(self, x):
        pass

    def _intersects_y(self, y):
        pass

class Pixel:

    def __init__(self, x, y):
        """
        (x,y): ABSOLUTE coordinates of TOP-LEFT corner of pixel.
        """
        self.x, self.y = x, y

        # RELATIVE coordinates of TOP-LEFT of pixel (relative to center of circle)
        self.dx =


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


# load phantom
filename = 'brain128.npy'
phantom = np.load(filename, 'r')
img_len = len(phantom)

#img = Image(n)

ps, Const.ps = 100, 100

coords = np.meshgrid(range(0, ps*img_len, ps), range(0, ps*img_len, ps))

print(np.array(coords).shape)





















pass
