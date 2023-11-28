def normalize(img, mean, std):
    img = (img - mean) / std
    return img
