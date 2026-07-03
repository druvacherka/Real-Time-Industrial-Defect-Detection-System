import torch


def get_device():
    """
    Returns the best available device.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def print_device_info():

    device = get_device()

    print("=" * 50)
    print("Device Information")
    print("=" * 50)

    print(f"Selected Device : {device}")

    if device.type == "cuda":
        print(f"GPU : {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version : {torch.version.cuda}")

    else:
        print("Running on CPU")