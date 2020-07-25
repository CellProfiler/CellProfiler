"""
/!\ WARNING - DEPRECATED /!\
utf16encode.py - encode unicode strings as escaped utf16
This is only used for pipeline version < 3 files
"""


def utf16decode(x):
    """Decode an escaped utf8-encoded string
    """
    y = ""
    state = -1
    for z in x:
        if state == -1:
            if z == "\\":
                state = 0
            else:
                y += str(z)
        elif state == 0:
            if z == "u":
                state = 1
                acc = ""
            else:
                y += str(z)
                state = -1
        elif state < 4:
            state += 1
            acc += z
        else:
            state = -1
            y += chr(int(acc + z, 16))
    return y
