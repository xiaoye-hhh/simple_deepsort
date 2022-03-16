import numpy as np

def tlwh_2_tlbr(tlwh:np.ndarray):
    """ 将[[x1,y1,w,h], [x1,y1,w,h]...]转换成 [x1,y1,x2,y2], [x1,y1,x2,y2]...]

    Args:
        tlwh (np.ndarray): Nx4的np数组
    
    Returns:
        _type_: Nx4的np数组
    """
    tlbr = tlwh.copy()
    tlbr[:, 2:] += tlbr[:, :2]

    return tlbr.astype(np.int32)


def xywh_2_tlbr(xywh:np.ndarray, img_size:tuple):
    """ 将[[x,y,w,h], [x,y,w,h]...]转换成 [x1,y1,x2,y2], [x1,y1,x2,y2]...]

    Args:
        xywh (np.ndarray): Nx4的np数组
        img_size: 图片大小 (img_h, img_w)
        inplace (bool, optional): 是否可以在原来的数组上操作. Defaults to False.

    Returns:
        _type_: Nx4的np数组
    """
    x, y, w, h = xywh
    tlbr = np.zeros_like(xywh)
    tlbr[:, 0] = max(x - w / 2, 0)
    tlbr[:, 1] = min(x + w / 2, img_size[1] - 1)
    tlbr[:, 2] = max(y - h / 2, 0)
    tlbr[:, 3] = min(y + h / 2, img_size[0] - 1)
    return tlbr.astype(np.int32)


def tlwh_2_xy(tlwh:np.ndarray):
    """ 将[[x,y,w,h], [x,y,w,h]...]转换成 [x, y], [x, y]...] """
    xy = np.zeros((tlwh.shape[0], 2), np.int32)
    xy[:, 0] = tlwh[:, 0] + tlwh[:, 2] // 2
    xy[:, 1] = tlwh[:, 1] + tlwh[:, 3] // 2
    return xy