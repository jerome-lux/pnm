# coding=utf-8
from numba import jit, float64, prange, int64
import numpy as np
#from scipy.spatial import cKDTree
from sklearn.neighbors import KDTree
from warnings import warn
from .Stats import distribution


def isinside_surface(coords, radius, extent, epsilon):

    maxr = coords + radius
    minr = coords - radius

    category = set()

    if np.all(maxr <= extent) and np.all(minr >= 0):
        if extent[0] - coords[0] - radius < epsilon:
            category.add('x+')
        if extent[1] - coords[1] - radius < epsilon:
            category.add('y+')
        if extent[2] - coords[2] - radius < epsilon:
            category.add('z+')
        if coords[0] - radius < epsilon:
            category.add('x-')
        if coords[1] - radius < epsilon:
            category.add('y-')
        if coords[2] - radius < epsilon:
            category.add('z-')

        if len(category) == 0:
            category.add('inner')

        return category

    return None


def isinside(coords, radius, extent, epsilon):

    maxr = coords + radius
    minr = coords - radius

    category = set()

    if np.all(maxr <= extent) and np.all(minr >= 0):
        if extent[0] - coords[0] < epsilon:
            category.add('x+')
        if extent[1] - coords[1] < epsilon:
            category.add('y+')
        if extent[2] - coords[2] < epsilon:
            category.add('z+')
        if coords[0] < epsilon:
            category.add('x-')
        if coords[1] < epsilon:
            category.add('y-')
        if coords[2] < epsilon:
            category.add('z-')

        if len(category) == 0:
            category.add('inner')

        return category

    warn("No Category affected to pore center{} with radius {} (min={},max={}".format(
        coords, radius, minr, maxr))
    return None


def eucl_dist(a, b):

    return np.einsum('ij,ij->i', a-b, a-b)**0.5


# 'float64(float64[:],float64[:])',
@jit('float64(float64[:],float64[:])', nopython=True, fastmath=True, nogil=True)
def eucl_dist_naive(c0, c1):

    "Return euclidean distance between two point in R3"

    distance = 0
    for i in range(3):
        distance = distance + (c0[i] - c1[i])*(c0[i] - c1[i])
    distance = distance**0.5

    return distance


@jit('int64[:](float64[:,:],float64[:,:])', nopython=True, fastmath=True, parallel=True, nogil=True)
def nearest_neighbor(c0, centers):
    """Return the nearsest neighbor index of point c0
    """

    ind = np.zeros((c0.shape[0])).astype(np.int64)

    for j in range(c0.shape[0]):

        mindistance = eucl_dist_naive(c0[j, :], centers[0, :])  # 1er point
        ind[j] = 0

        for i in range(1, centers.shape[0]):
            distance = eucl_dist_naive(c0[j, :], centers[i, :])
            if distance < mindistance:
                mindistance = distance
                ind[j] = i

    return ind

# The Fastest !


# 'float64(float64[:],float64[:,:],float64[:])',
@jit('float64(float64[:],float64[:,:],float64[:])', nopython=True, fastmath=True, parallel=True, nogil=True)
def query_mindist_loops(c0, centers, radii):

    """Retourne la distance min entre la coord c0 et N autres points
    Méthode a priori la plus rapide """

    mindistance = eucl_dist_naive(c0, centers[0, :]) - radii[0]  # 1er point

    for i in range(1, centers.shape[0]):
        distance = eucl_dist_naive(c0, centers[i, :]) - radii[i]
        if distance < mindistance:
            mindistance = distance

    return mindistance


# 'float64(float64[:],float64[:,:],float64[:])',
@jit(nopython=True, fastmath=True, parallel=True, nogil=True)
def query_mindist_loops_parallel(c0, centers, radii):

    """Retourne la distance min entre la coord c0 et N autres points """

    distances = np.zeros_like(radii)

    for i in prange(0, centers.shape[0]):
        distances[i] = eucl_dist_naive(c0, centers[i, :]) - radii[i]

    mindist = distances[0]
    for d in distances:
        if d < mindist:
            mindist = d
    return mindist


def generate_radii(distributions_props, nbnodes):

    """Return nbnodes values following the distributions given in distributions_props
    """

    radii = []

    for props, probs in distributions_props:
        distgenerator = props.get("func", distribution)
        r = list(distgenerator(n_samples=int(np.ceil(probs*nbnodes)), **props))
        radii.extend(r)

    return np.array(radii)[0:nbnodes]