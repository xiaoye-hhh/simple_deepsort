import numpy as np
import torch
from deal_coordinate import tlwh_2_tlbr, tlwh_2_xy
from calc_cost_matrix import calc_feature_cost_matrix, calc_distance_matrix
from match import match
from tracker import Tracker
from model import FeatureExtractor

class TrackerManager:
    def __init__(self,  max_feature, max_ttl, min_age, threshold, tolerate_range, distance_base_cost) -> None:
        self.extractor = FeatureExtractor((256, 128))
        self.max_feature = max_feature
        self.max_ttl = max_ttl
        self.min_age = min_age
        self.tolerate_range = tolerate_range
        self.distance_base_cost = distance_base_cost
        self.threshold = threshold
        self.trackers = []
        self.mean_xy = np.zeros(2)


    @torch.no_grad()
    def update(self, boxes, ori_img):
        ''' 
            0. 增加一个时间单位
            1. 抽取特征
            2. 计算距离矩阵
            3. 匹配
            4. 更新
            5. 构造返回值
        '''
        self.__increase_trackers_ttl_and_age()
        if len(boxes) == 0:
            return np.zeros((0, 5))

        # 这里需要修改
        box_tlbr = tlwh_2_tlbr(boxes)
        box_xy = tlwh_2_xy(boxes)

        feature_short, feature_long = self.__get_detections_features(box_tlbr, ori_img)
        feature_cost_matrix = calc_feature_cost_matrix(self.trackers, feature_long)
        distance_cost_matrix, self.mean_xy = calc_distance_matrix(self.trackers, box_xy, self.mean_xy, self.tolerate_range, self.distance_base_cost)
        matched, unmatched_trackers, unmatched_detections = match(feature_cost_matrix + distance_cost_matrix, self.threshold)

        ret = []
        for tracker_indice, detection_indice in matched:
            tracker = self.trackers[tracker_indice]
            tracker.update(box_xy[detection_indice], feature_short[detection_indice], feature_long[detection_indice])
            if tracker.is_online():
                ret.append(np.concatenate([box_tlbr[detection_indice], np.array([tracker.id])]))

        for tracker_indice in reversed(unmatched_trackers):
            if self.trackers[tracker_indice].is_dead():
                self.trackers.pop(tracker_indice)

        for detection_indice in unmatched_detections:
            self.trackers.append(Tracker(box_xy[detection_indice], feature_short[detection_indice], feature_long[detection_indice], self.max_feature, self.max_ttl, self.min_age))

        return np.asarray(ret)


    def __get_detections_features(self, boxes:np.ndarray, ori_img:np.ndarray):
        """ 获取图片的特征
                - 由box载ori_img上框选目标
                - 送入特征提取器
        Args:
            boxes (np.ndarray): 检测框
            ori_img (np.ndarray): 图片

        Returns:
            _type_: features, torch.Tensor (cuda上) [NxD]
        """
        im_crops = []

        boxes[:, 0] = np.maximum(boxes[:, 0], 0)
        boxes[:, 1] = np.maximum(boxes[:, 1], 0)
        boxes[:, 2] = np.minimum(boxes[:, 2], ori_img.shape[1])
        boxes[:, 3] = np.minimum(boxes[:, 3], ori_img.shape[0])

        for box in boxes:
            x1, y1, x2, y2 = box
            im = ori_img[y1:y2, x1:x2, :]
            im_crops.append(im)

        return self.extractor(im_crops)


    def __increase_trackers_ttl_and_age(self):
        ''' 增加一个时间单位 '''
        for tracker in self.trackers:
            tracker.increase_ttl_and_age()