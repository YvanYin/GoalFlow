from typing import Union, List, Tuple, Any, Optional

__all__ = ["list_sum", "val2list", "squeeze_list", "make_divisible", "get_same_padding"]


def list_sum(x: List) -> Any:
    """Return the sum of a list of objects.

    can be int, float, torch.Tensor, np.ndarray, etc
    can be used for adding losses
    """
    return x[0] if len(x) == 1 else x[0] + list_sum(x[1:])


def val2list(val: Union[List, Tuple, Any], repeat_time=1) -> List:
    """Repeat `val` for `repeat_time` times and return the list or val if list/tuple."""
    if isinstance(val, (list, tuple)):
        return list(val)
    return [val for _ in range(repeat_time)]


def squeeze_list(src_list: Optional[List]) -> Union[List, Any]:
    """Return the first item of the given list if the list only contains one item.

    usually used in args parsing
    """
    if src_list is not None and len(src_list) == 1:
        return src_list[0]
    else:
        return src_list


def make_divisible(v: Union[int, float], divisor: Optional[int], min_val=None) -> Union[int, float]:
    """This function is taken from the original tf repo.

    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_val:
    :return:
    """
    if divisor is None:
        return v

    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def get_same_padding(kernel_size: Union[int, Tuple[int, int]]) -> Union[int, tuple]:
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, f"invalid kernel size: {kernel_size}"
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    else:
        assert isinstance(kernel_size, int), "kernel size should be either `int` or `tuple`"
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2
