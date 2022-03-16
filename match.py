import numpy as np
from scipy.optimize import linear_sum_assignment


def match(cost_matrix: np.ndarray, threshold: float):
    """ 用匈牙利算法作匹配: 只保留小于阈值的匹配

    Args:
        cost_matrix (np.ndarray): trackers x detections 的费用矩阵
        threshold (float): 接受匹配的阈值

    Returns:
        _type_: matched_indices: [[tracker_indice, detection_indice], [tracker_indice, detection_indice]...]
                unmatched_trackers: list[int]
                unmatched_detections: list[int]
    """
    N, M = cost_matrix.shape
    if N == 0 or M == 0:
        return np.zeros((0, 2)), list(range(N)), list(range(M))

    # 匈牙利算法作匹配
    cost_matrix = np.minimum(cost_matrix, threshold)
    candidate_matched_indices = np.array(linear_sum_assignment(cost_matrix)).T

    # 只选取cost小于threshold的匹配
    matched_indices = []
    for inidice in candidate_matched_indices:
        if cost_matrix[inidice[0], inidice[1]] < threshold:
            matched_indices.append(inidice)
    matched_indices = np.array(matched_indices) if len(matched_indices) else np.zeros((0, 2))

    unmatched_trackers = []
    for i in range(cost_matrix.shape[0]):
        if i not in matched_indices[:, 0]:
            unmatched_trackers.append(i)

    unmatched_detections = []
    for i in range(cost_matrix.shape[1]):
        if i not in matched_indices[:, 1]:
            unmatched_detections.append(i)

    return matched_indices, unmatched_trackers, unmatched_detections