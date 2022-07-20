"""normalize the data to binary from 0 and 255, to 0's and 1's"""
import numpy


def normalize(data):
    """normalize the data between 0 and 1 from (0,255) image format

    Args:
        data (_type_): data to normalize between 0 and 1

    Returns:
        _type_: _description_
    """
    return data / 255.0
