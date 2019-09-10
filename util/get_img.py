import os
def get_img(path):
    imgs = []
    for img in os.listdir(path):
        imgs.append(os.path.join(path, img))
    return imgs