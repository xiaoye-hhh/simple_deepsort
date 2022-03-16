import torch
import numpy as np
from tracker import Tracker

@torch.no_grad()
def calc_feature_cost_matrix(trackers: list[Tracker], detection_features: torch.Tensor, device='cuda'):
    ''' 计算距离: 只计算长维度的特征 '''
    cost_matrix = torch.zeros((len(trackers), len(detection_features)), device=device)
    for i, tracker in enumerate(trackers):
        _, tracker_features = tracker.get_feature()
        cost = tracker_features @ detection_features.T
        cost = 1 - cost.max(dim=0)[0]
        cost_matrix[i] = cost
    
    return cost_matrix.cpu().numpy()

def calc_distance_matrix(trackers: list[Tracker], detection_xy: np.ndarray, last_xy: np.ndarray, tolerate_range: float, base_cost=0.5):
    '''
        - 判断是否进行了大的移动, 如果是, 不考虑
        - 计算距离
    '''
    N, M = len(trackers), len(detection_xy)
    detection_mean_xy = detection_xy.mean(axis=0)
    if np.abs(detection_mean_xy - last_xy).mean() > tolerate_range or len(detection_xy) == 0 or len(trackers) == 0:
        print('false', np.abs(detection_mean_xy - last_xy).mean())
        return np.zeros((N, M)), detection_mean_xy
    
    # 将tracker中的信息统计起来
    tracker_xy = np.zeros((N, 1, 2))
    for i, tracker in enumerate(trackers):
        tracker_xy[i, 0] = tracker.xy 
    tracker_xy.repeat(M, 1)

    # 根据前后两帧的偏离距离计算
    distance_matrix = np.abs((tracker_xy - detection_xy)).sum(axis=2)
    cost_matrix = np.full((N, M), base_cost)
    cost_matrix[distance_matrix < tolerate_range] = 0

    # 对于那些丢失追踪的需要打一定折扣
    for i, tracker in enumerate(trackers):
        if tracker.is_online():
            cost_matrix[i] *= (60 - min(30, tracker.ttl)) * 0.1667
        else:
            cost_matrix[i] += base_cost 

    return cost_matrix, detection_mean_xy



