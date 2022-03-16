import torch


class Tracker:
    cnt = 0

    def __init__(self, xy, feature_short, feature_long, max_feature, max_ttl=90, min_age=5) -> None:
        self.id = Tracker.cnt
        Tracker.cnt += 1

        self.xy = xy 
        self.age = 0
        self.ttl = 0
        self.max_ttl = max_ttl
        self.min_age = min_age
        
        self.feature_full = False
        self.feature_index = 1
        self.max_feature = max_feature
        self.features_long = torch.zeros((max_feature, feature_long.shape[0]), device='cuda')
        self.features_short = torch.zeros((max_feature, feature_short.shape[0]), device='cuda')
        self.features_long[0] = feature_long
        self.features_short[0] = feature_short

    
    def update(self, xy, feature_short, feature_long):
        ''' 更新信息: 时间信息, 位置信息, 特征信息 '''
        self.ttl = 0
        self.xy = xy        
        self.features_short[self.feature_index] = feature_short
        self.features_long[self.feature_index] = feature_long
        self.feature_index += 1
        if self.feature_index == self.max_feature:
            self.feature_full = True
            self.feature_index = 0
    
    
    def get_feature(self):
        ''' 返回所有的特征向量: 短的 长的 '''
        if self.feature_full:
            return self.features_short, self.features_long
        else:
            return self.features_short[:self.feature_index], self.features_long[:self.feature_index]
            

    def is_dead(self):
        ''' 超出最大存活时间就丢弃 or 小于最小的存活时间 '''
        return self.ttl >= self.max_ttl or self.age <= self.min_age


    def is_online(self):
        ''' age超过min_age就是有用的追踪信息 '''
        return self.age > self.min_age


    def increase_ttl_and_age(self):
        ''' 增加ttl和age '''
        self.ttl += 1
        self.age += 1
    