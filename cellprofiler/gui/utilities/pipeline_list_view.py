import wx


def plv_get_bitmap(data):
    return wx.Bitmap(data)


def get_image_index(name):
    """Return the index of an image in the image list"""
    global image_index_dictionary
    if name not in image_index_dictionary:
        image_index_dictionary[name] = len(image_index_dictionary)
    return image_index_dictionary[name]
