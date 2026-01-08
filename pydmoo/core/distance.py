import numpy as np
from scipy.linalg import logm, sqrtm


def wasserstein_distance(mean1, cov1, mean2, cov2):
    diff = mean1 - mean2
    sqrt_cov1 = sqrtm(cov1)
    cross_term = sqrtm(sqrt_cov1 @ cov2 @ sqrt_cov1)
    return np.sqrt(diff @ diff + np.trace(cov1 + cov2 - 2 * cross_term))


def kl_div_distance(mean1, cov1, mean2, cov2):
    inv_cov2 = np.linalg.inv(cov2)
    diff = mean2 - mean1
    logdet = np.log(np.linalg.det(cov2) / np.linalg.det(cov1))
    return 0.5 * (np.trace(inv_cov2 @ cov1) + diff.T @ inv_cov2 @ diff - len(mean1) + logdet)


def bhattacharyya_distance(mean1, cov1, mean2, cov2):
    avg_cov = 0.5 * (cov1 + cov2)
    inv_avg = np.linalg.inv(avg_cov)
    diff = mean1 - mean2
    return 0.125 * diff.T @ inv_avg @ diff + 0.5 * np.log(
        np.linalg.det(avg_cov) / np.sqrt(np.linalg.det(cov1) * np.linalg.det(cov2))
    )


def mahalanobis_distance(mean1, cov1, mean2, cov2):
    inv_cov = np.linalg.inv(0.5 * (cov1 + cov2))
    return np.sqrt((mean1 - mean2).T @ inv_cov @ (mean1 - mean2))


def riemannian_distance(cov1, cov2):
    return np.linalg.norm(logm(np.linalg.inv(sqrtm(cov1)) @ cov2 @ np.linalg.inv(sqrtm(cov1))), "fro")


def frobenius_distance(cov1, cov2):
    return np.linalg.norm(cov1 - cov2, "fro")


# Used in NSGA2-IMKT-N
# def norm_mean_frobenius_distance(mean, cov, target_mean, target_cov):
#     return np.linalg.norm(mean - target_mean) + frobenius_distance(cov, target_cov)
def norm_mean_frobenius_distance(mean, cov, target_mean, target_cov):
    return frobenius_distance(cov, target_cov)
