import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave

from numpy.linalg import eig, det
from scipy.linalg import lstsq

from skimage.morphology import watershed
from skimage.measure import regionprops

arctan = np.arctan
sqrt = np.sqrt
cos = np.cos
sin = np.sin
pi = np.pi

from patches import make_patches

class ParametricEllipse(object):
    def __init__(self, h, k, a, b, τ):
        self.h = h
        self.k = k
        self.a = a
        self.b = b
        self.τ = τ
    def decompose(self):
        return (self.h, self.k, self.a, self.b, self.τ)
    def get_points(self):
        (h, k, a, b, τ) = self.decompose()
        t = np.linspace(0, 2*np.pi, 1000)
        xs_untrans = a * cos(t)
        ys_untrans = b * sin(t)
        rotation_matrix = np.array([
            [np.cos(τ), -np.sin(τ)],
            [np.sin(τ),  np.cos(τ)]
        ])
        untrans = np.concatenate([[xs_untrans, ys_untrans]])
        rotated = rotation_matrix @ untrans
        return rotated + np.array([[h, k]]).repeat(1000, axis=0).T
    def get_skeleton_points(self):
        (h, k, a, b, τ) = self.decompose()
        rotation_matrix = np.array([
            [np.cos(τ), -np.sin(τ)],
            [np.sin(τ),  np.cos(τ)]
        ])

        zeros = np.zeros(500)
        point_xs = np.concatenate([np.linspace(-a, a, 500), zeros])
        point_ys = np.concatenate([zeros, np.linspace(-b, b, 500)])
        untrans = np.concatenate([[point_xs, point_ys]])
        rotated = rotation_matrix @ untrans
        return rotated + np.array([[h, k]]).repeat(1000, axis=0).T
    def to_conic(self):
        h, k, a, b, τ = self.decompose()
        c, s = cos(τ), sin(τ)
        A = (b*c)**2 + (a*s)**2
        B = -2*c*s*(a**2 - b**2)
        C = (b*s)**2 + (a*c)**2
        D = -2*A*h - k*B
        E = -2*C*k - h*B
        F = -(a*b)**2 + A*h**2 + B*h*k + C*k**2
        return ConicEllipse(B/A, C/A, D/A, E/A, F/A)

class ConicEllipse(object):
    def __init__(self, B, C, D, E, F):
        assert B**2 - 4*C < 0
        assert D**2 / 4 + E**2 / (4*C) - F > 0
        self.A = 1
        self.B = B
        self.C = C
        self.D = D
        self.E = E
        self.F = F
    def decompose(self):
        return (self.A, self.B, self.C, self.D, self.E, self.F)
    def fit(xs, ys):
        abscissas = np.concatenate([[xs**2, xs*ys, ys**2, xs, ys]]).T
        ordinates = -np.ones(len(abscissas))
        c, residues, rank, s = lstsq(abscissas, ordinates)
        B = c[1] / c[0]
        C = c[2] / c[0]
        D = c[3] / c[0]
        E = c[4] / c[0]
        F = 1 / c[0]
        return ConicEllipse(B, C, D, E, F)
    def to_parametric(self):
        A, B, C, D, E, F = self.decompose()
        M0 = np.array([
            [F, D/2, E/2],
            [D/2, A, B/2],
            [E/2, B/2, C]
        ])
        M = np.array([
            [A, B/2],
            [B/2, C]
        ])
        [e1, e2] = eig(M)[0]
        if abs(e1 - A) <= abs(e1 - C):
            λ1, λ2 = e1, e2
        else:
            λ1, λ2 = e2, e1
        a = sqrt(-det(M0)/(det(M)*λ1))
        b = sqrt(-det(M0)/(det(M)*λ2))
        h = (B*E - 2*C*D)/(4*A*C - B**2)
        k = (B*D - 2*A*E)/(4*A*C - B**2)
        τ = arctan(B/(A-C)) / 2
        return ParametricEllipse(h, k, a, b, τ)

def get_probability_map(img, clf):
    """
    Get the probability map for a given image
    """
    patch_radius = 3
    patch_size = 2 * patch_radius + 1
    patches = make_patches(img, patch_radius)
    row_patch_count = patches.shape[0]
    col_patch_count = patches.shape[1]
    fuzzy_edge = np.zeros(img.shape, dtype=float)
    inputs = patches.reshape((row_patch_count * col_patch_count, patch_size ** 2))
    outputs = clf.predict_proba(inputs)[:, 1].reshape((row_patch_count, col_patch_count))
    fuzzy_edge[patch_radius:-patch_radius, patch_radius:-patch_radius] = outputs
    return fuzzy_edge

def get_orientation(probability_map, method='ellipse') -> float:
    """
    Get the orientation from the ellipse with a given probability map

    Methods
    -------
    Ellipse:
        This collects 'potential edge points' by selecting all points where >20%
        of trees classified them as edges. It then does a least-squares fit to
        the general conic section formula, throwing an exception if the section
        is not elliptical. It then converts that to a parametric-form ellipse
        and extracts the orientation.
    Blob:
        This performs watershed segmentation on the conic section formula, then
        calculates the orientation on the resulting blob by using the ellipse
        with matching second moments to the binary blob.
    """
    if method == 'ellipse':
        ys, xs = (probability_map > 0.2).nonzero()
        conic_ellipse = ConicEllipse.fit(xs, ys)
        parametric_ellipse = conic_ellipse.to_parametric()
        return -parametric_ellipse.τ
    elif method == 'blob':
        labeled = watershed(probability_map, 2)
        blob = regionprops(labeled)[0]
        return blob.orientation
    else:
        raise ValueError("Invalid method: must be ellipse or blob")
