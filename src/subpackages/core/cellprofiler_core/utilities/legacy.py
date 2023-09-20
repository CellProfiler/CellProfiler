"""
Functions associated with legacy python
"""


def cmp(a, b):
    return int((a > b)) - int((a < b))


def equals(a, b, encoding_a="utf-8", encoding_b="utf-8"):
    """
    Purpose: Used primarily for equality comparisons between bytes and str but can be used for any type

    :param a: any python type, including bytes
    :param b: any python type, including bytes
    :param encoding_a: str designating encoding format [default="utf-8"]
    :param encoding_b: str designating encoding format [default="utf-8"]
    :return: bool
    """
    if isinstance(a, bytes):
        a = a.decode(encoding_a)
    if isinstance(b, bytes):
        b = b.decode(encoding_b)
    return a == b


def convert_bytes_to_str(a, encoding="utf-8"):
    """
    Purpose: Converts the byte objects in the 1-D container into str (does not work for nested elements)
    :param a: iterable 1-D container
    :param encoding: str designating encoding format [default="utf-8"]
    :return: 1-D container with bytes converted to str
    """
    a_type = type(a)
    return a_type(map(lambda x: x.decode(encoding) if isinstance(x, bytes) else x, a))
