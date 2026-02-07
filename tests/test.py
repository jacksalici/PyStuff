from pystuff.logger import Logger
from pystuff.shapehook import ShapeHook

import torch
from torch import nn


def test_model_output():
    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 16 * 16, 10)
    )

    dummy_input = torch.randn(1, 3, 32, 32)
    output = model(dummy_input)

    assert output.shape == (1, 10), f"Unexpected output shape: {output.shape}"
    print("Model output test passed.")


def test_logger():
    logger = Logger(print_log_level=Logger.LogLevel.INFO)
    logger.log({"message": "Testing logger at INFO level"}, log_level=Logger.LogLevel.INFO)
    logger.log({"message": "Testing logger at DEBUG level"}, log_level=Logger.LogLevel.DEBUG)
    print("Logger test passed.")


def test_shapehook():
    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 16 * 16, 10)
    )

    hook_manager = ShapeHook()
    hook_manager.register_hooks(model, one_time=True)

    dummy_input = torch.randn(1, 3, 32, 32)
    _ = model(dummy_input)  # Forward pass to trigger hooks

    print("ShapeHook test passed.")


if __name__ == "__main__":
    test_model_output()
    test_logger()
    test_shapehook()

