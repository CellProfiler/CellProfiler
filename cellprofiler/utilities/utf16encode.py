"""
/!\ WARNING - DEPRECATED /!\
utf16encode.py - encode unicode strings as escaped utf16
This is only used for pipeline version < 3 files
"""

import six


def utf16decode(x):
    """Decode an escaped utf8-encoded string
    """
    y = u""
    state = -1
    for z in x:
        if state == -1:
            if z == "\\":
                state = 0
            else:
                y += six.text_type(z)
        elif state == 0:
            if z == "u":
                state = 1
                acc = ""
            else:
                y += six.text_type(z)
                state = -1
        elif state < 4:
            state += 1
            acc += z
        else:
            state = -1
            y += six.unichr(int(acc + z, 16))
    return y
