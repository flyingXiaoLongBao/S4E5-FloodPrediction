import torch


# 需要在模型、损失函数、当前批次的数据后加入to(device)
def get_available_device():
    """
    获取可用的设备
    
    Returns:
        torch.device: 可用的设备对象
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')  # 苹果Metal
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():  # 检查英特尔XPU
        return torch.device('xpu')
    else:
        return torch.device('cpu')
