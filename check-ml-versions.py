import sys
import platform
import pandas as pd
import sklearn as sk
import torch
import tensorflow as tf
import tensorflow.keras as keras

def check_versions():
    specs = [
        ("Python Platform", platform.platform()),
        ("Python", sys.version.split(" ")[0]),
        ("TensorFlow Version", tf.__version__),
        ("Keras Version", keras.__version__),
        ("Pandas", pd.__version__),
        ("Scikit-Learn", sk.__version__),
        ("PyTorch", torch.__version__)
    ]

    header = "{:<20} {:<15}".format("Library", "Version")
    print(header)
    print("-" * len(header))

    for spec in specs:
        print("{:<20} {:<15}".format(*spec))

    gpu = len(tf.config.list_physical_devices('GPU')) > 0
    print("\nTensorFlow GPU support:", "available" if gpu else "NOT AVAILABLE")
    print("Note: 'available' indicates TensorFlow is using GPU.")

    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print("\nPyTorch MPS support:", "available")
        print("MPS device found:", x)
    else:
        print("\nPyTorch MPS support:", "NOT AVAILABLE")
        print("Note: 'NOT AVAILABLE' indicates PyTorch is not using Metal Performance Shaders (MPS) on Mac.")

if __name__ == "__main__":
    check_versions()

