# encoding : utf-8

from math import sqrt
import numpy as np
from PIL import Image
from time import strftime, time

def load_image( image_path ):
    # try:
    image = Image.open(image_path)
    image = np.array(image)
    return image
    # except:
        # print('open', image_path, 'error')

def judge( x ):
    if x < 64 : return 0
    elif x < 128: return 1
    elif x < 192: return 2
    else: return 3

def deal_image(id, image, vectors):
    H, W, D = image.shape
    pos = 0
    for h in range(0, H, 2):
        for w in range(0, W, 2):
            if h+2 > H or w+2 > W : h, w= H-2, W-2
            vectors[id, pos, 0] = np.mean(image[h:h+2, w:w+2, 0])
            vectors[id, pos, 1] = np.mean(image[h:h+2, w:w+2, 1]) 
            vectors[id, pos, 2] = np.mean(image[h:h+2, w:w+2, 2]) 
            pos += 1

def cos_similar(x, y):
    # xx, yy, xy = 0.0, 0.0, 0.0
    ret = 0.0
    n = x.shape[0]
    for i in range(n):
        x0 = max(abs(x[i, 0] - y[i, 0]) - 10, 0)
        x1 = max(abs(x[i, 1] - y[i, 1]) - 10, 0)
        x2 = max(abs(x[i, 2] - y[i, 2]) - 10, 0)
        ret += x0 * x0
        ret += x1 * x1
        ret += x2 * x2
    return ret

def deal_image_repo( repo_path, block, batch ):
    n = block[0] * block[1] // 4
    vectors = np.zeros((batch, n, 3), dtype=int)
    images = []
    for i in range(batch):
        image = Image.open(repo_path + '/repo'+str(i)+'.jpg')
        image = np.array(image.resize(block), dtype=int)
        images.append(image)
        deal_image(i, image, vectors)
    return vectors, images


def replace( image, index, y, images, block, batch ):
    n = block[0] * block[1] // 4
    x = np.zeros((1, n, 3))
    deal_image(0, image[ index[0] : index[1],
                         index[2] : index[3], : ], x)
    max_cos, idx = 1e25, 0
    for i in range(batch):
        temp = cos_similar(x[0, :, :], y[i, :, :])
        if temp < max_cos:
            max_cos = temp
            idx = i
    # print( images[idx].shape,  index[0], index[1], index[2], index[3])
    image[ index[0] : index[1],
           index[2] : index[3], : ] =  images[idx]

def split_image( image_path, block, batch ):
    image = load_image(image_path)
    # print(image.shape)
    h, w = image.shape[:2]
    y, images = deal_image_repo('./image', block, batch)
    for i in range(0, h, block[0]):
        for j in range(0, w, block[1]):
            if i + block[0] > h: continue
            if j + block[1] > w: continue
            index = ( i, i + block[0], 
                      j, j + block[1])
            replace(image, index, y, images, block, batch)
    image = np.array(image, dtype=np.uint8)
    image = Image.fromarray(image)
    image.show()
    return image
    

if __name__ == '__main__':
    start = time()
    print(strftime('[%H:%M:%S]'))
    split_image('test.jpg', (10,10), 50)
    print(strftime('[%H:%M:%S]'))
    print(time()-start)
#     image = load_image('./img.jpg')
