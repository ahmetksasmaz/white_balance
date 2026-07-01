def get_torch_device(use_gpu=False):
    if not use_gpu:
        return 'cpu'
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        if torch.backends.mps.is_available():
            return 'mps'
    except Exception:
        pass
    return 'cpu'
