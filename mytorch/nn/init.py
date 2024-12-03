# mytorch/nn/init.py


def calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out cannot be computed for tensor with fewer than 2 dimensions")
    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        receptive_field_size = tensor[0][0].numel()
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def kaiming_uniform(tensor, a=5**0.5):
    fan_in, fan_out = calculate_fan_in_and_fan_out(tensor)
    bound = (6 / fan_in) ** 0.5 if fan_in > 0 else 0
    tensor.data.uniform_(-bound, bound)
