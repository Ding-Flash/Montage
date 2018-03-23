# encoding : utf-8

from math import sqrt
import numpy as np
from PIL import Image

def load_image( image_path ):
    try:
        image = Image.open(image_path)
        image = np.array()
        return image
    except:
        print('open', image_path, 'error')

def judge( x ):
    if x < 64 : return 0
    elif x < 128: return 1
    elif x < 192: return 2
    else: return 3

def deal_image(id, image, vectors):
    H, W, D = image.shape
    for h in range(H):
        for w in range(W):
            r, g, b = image[h, w, :]
            r = judge(r)
            g = judge(g)
            b = judge(b)
            vectors[id, 16 * r + 4 * g + b] += 1

def cos_similar(x, y):
    xx, yy, xy = 0.0, 0.0, 0.0
    for i in range(64):
        xx += x[i] * x[i]
        yy += y[i] * y[i]
        xy += x[i] * y[i]
    return xy / (sqrt(xx) * sqrt(yy))

def deal_image_repo( repo_path, block ):
    vectors = np.zeros((20, 64), dtype=int)
    images = []
    for i in range(20):
        image = Image.open(repo_path + '/repo'+str(i)+'.jpg')
        image = np.array(image.resize(block), dtype=int)
        images.append(image)
        deal_image(i, image, vectors)
    return vectors, images


def replace( image, index, block ):
    x = np.zeros((1, 64))
    deal_image(0, image[ index[0] : index[1],
                         index[2] : index[3], : ], x)
    y, images = deal_image_repo('./image', block)
    max_cos, idx = 0.0, 0
    for i in range(20):
        temp = cos_similar(x[0, :], y[i, :])
        if temp > max_cos:
            max_cos = temp
            idx = i
    print( images[idx].shape,  index[0], index[1], index[2], index[3])
    image[ index[0] : index[1],
           index[2] : index[3], : ] =  images[idx]

def split_image( image_path, block ):
    image = load_image(image_path)
    print(image.shape)
    h, w = image.shape[:2]
    for i in range(0, h, block[0]):
        for j in range(0, w, block[1]):
            if i + block[0] >= h: continue
            if j + block[1] >= w: continue
            index = ( i, i + block[0], 
                      j, j + block[1])
            replace(image, index, block)
    image = np.array(image, dtype=np.uint8)
    image = Image.fromarray(image)
    image.show()
    return image
    

# if __name__ == '__main__':
    
    
#     image = load_image('./img.jpg')
