import numpy as np
from scipy import linalg
import matplotlib.image as img
from matplotlib import pyplot as plt

def image_to_matrix(image_path):
    image = img.imread(image_path)
    image_matrix = np.array(image)
    return image_matrix

def zero_diagonal(matrix, eps):
    diagonal = np.diag(matrix)
    c = np.zeros(abs(matrix.shape[0] -  matrix.shape[1]))
    diagonal = np.concatenate((diagonal, c), axis = None)
    mask = abs(diagonal) < eps
    rank = matrix.shape[0] - np.count_nonzero(mask)
    matrix[mask[:matrix.shape[0]], :] = 0
    return matrix, rank

def invert_permutation(p):
    p = np.asanyarray(p)
    s = np.empty_like(p)
    s[p] = np.arange(p.size)
    return s

image_matrix = np.mean(image_to_matrix('im3.jpeg'), axis=2)
U, E, V = np.linalg.svd(image_matrix)
Q, R, P = linalg.qr(image_matrix, pivoting=True)

for i in range(0, 6):
    mask = (E != E)
    rank = E.shape[0] // (2 ** i)
    for j in range(E.shape[0] - 1, rank - 1, -1):
        mask[j] = True
    E_1 = E
    E_1[mask] = 0
    cut_img_matr_svd = np.dot(U[:, :E_1.shape[0]] * E_1, V[:E_1.shape[0]])
    er = np.linalg.norm(cut_img_matr_svd - image_matrix) / np.linalg.norm(image_matrix)
    plt.subplot(2, 3, i+1)
    plt.imshow(cut_img_matr_svd, cmap='gray')
    plt.title(f'rank = {rank}, $ \| A - B \| / \|B\| $ =' + '{0:.2e}'.format(er)) 
plt.show()

for i in range(0, 6):
    mask = np.zeros((R.shape[0],), dtype = bool)
    rank = R.shape[0] // (2 ** i)
    for j in range(R.shape[0] - 1, rank - 1, -1):
        mask[j] = True
    R_1 = R
    R_1[mask, :] = 0
    cut_img_matr_qr = np.dot(Q, R_1)[:, invert_permutation(P)]
    er = np.linalg.norm(cut_img_matr_qr - image_matrix) / np.linalg.norm(image_matrix)
    plt.subplot(2, 3, i+1)
    plt.imshow(cut_img_matr_qr, cmap='gray')
    plt.title(f'rank = {rank}, $ \| A - B \| / \|B\| $ =' + '{0:.2e}'.format(er)) 
plt.show()