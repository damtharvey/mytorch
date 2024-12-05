from typing import Tuple
import torch


def calculate_fan_in_and_fan_out(tensor: torch.Tensor) -> Tuple[int, int]:
    """
    Computes the fan-in and fan-out values for a given tensor.

    Fan-in is the number of input features, and fan-out is the number of output features.
    These values are used for initializing weights in neural networks.

    Args:
        tensor (torch.Tensor): The tensor for which fan-in and fan-out need to be calculated.

    Returns:
        Tuple[int, int]: A tuple containing fan-in and fan-out values.

    Raises:
        ValueError: If the tensor has fewer than 2 dimensions.
    """
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out cannot be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1

    # For convolutional layers, compute the receptive field size
    if tensor.dim() > 2:
        receptive_field_size = tensor[0][0].numel()

    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def kaiming_uniform(tensor: torch.Tensor, a: float = 5**0.5) -> None:
    """
    Initializes a tensor with values sampled from a uniform distribution.

    The distribution is determined by the Kaiming initialization method,
    which ensures the variance of the weights remains constant across layers.

    Args:
        tensor (torch.Tensor): The tensor to initialize.
        a (float): A scaling factor (default is sqrt(5)).

    Returns:
        None: The tensor is modified in-place.
    """
    fan_in, _ = calculate_fan_in_and_fan_out(tensor)
    bound = (6 / fan_in) ** 0.5 if fan_in > 0 else 0
    tensor.data.uniform_(-bound, bound)
