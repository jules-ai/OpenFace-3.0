import torch
import cv2
import numpy as np
import argparse
import math
import copy

# For building the standalone demo, we need to import the network building logic from STAR.
# Since it relies on the openface-test package (which contains STAR modules), we import it here:
from openface.STAR.lib import utility

class AlignmentModelWrapper(torch.nn.Module):
    """
    A simple wrapper for the STAR alignment model to cleanly handle input/output
    for tracing/ONNX export.
    """
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        # The STAR net outputs a list/tuple of intermediate representations.
        # The final landmarks are usually the last element of the output,
        # and we only need the first element of that tuple (the normalized landmarks coordinates)
        output = self.net(x)
        # Assuming output[-1][0] contains the normalized points of shape [batch, num_points, 2]
        return output[-1][0]


def get_star_model(model_path, config_args):
    """
    Loads the STAR model (e.g. 98 point WFLW model) into a clean PyTorch module.
    """
    config = utility.get_config(config_args)
    # Prevent logger initialization issues during simple inference/export
    config.logger = None

    net = utility.get_net(config)
    checkpoint = torch.load(model_path, map_location="cpu")
    net.load_state_dict(checkpoint["net"])
    net.eval()

    return AlignmentModelWrapper(net)

def export_to_onnx(model, dummy_input, onnx_path):
    print(f"Exporting model to ONNX: {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input_image'],
        output_names=['landmarks'],
        dynamic_axes={'input_image': {0: 'batch_size'}, 'landmarks': {0: 'batch_size'}}
    )
    print("Export successful!")

if __name__ == '__main__':
    # 1. Configuration for 98-point WFLW model
    args = argparse.Namespace()
    args.config_name = 'alignment'

    model_path = './weights/Landmark_98.pkl' # Or the downloaded WFLW checkpoint

    print("Loading PyTorch model...")
    try:
        model = get_star_model(model_path, args)
    except Exception as e:
        print(f"Failed to load model: {e}. Please ensure the weights exist at {model_path} and the openface-test pip package is installed.")
        exit(1)

    print("Model loaded successfully!")

    # 2. Define dummy input (Batch size 1, 3 channels, 256x256 image)
    dummy_input = torch.randn(1, 3, 256, 256)

    # 3. Test Inference
    print("Running PyTorch dummy inference...")
    with torch.no_grad():
        out = model(dummy_input)
    print(f"Output shape (normalized points): {out.shape}")

    # 4. Export to ONNX
    export_to_onnx(model, dummy_input, "STAR_98_landmarks.onnx")
