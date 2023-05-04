import os
import random
import shutil
import Augmentor
from itertools import combinations
import imutils
import numpy as np
import cv2 as cv
from PIL import Image
import glob
from natsort import natsorted
from os import listdir
import imageio
import imgaug as ia
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug import augmenters as iaa
#ia.seed(1)

# count: 1 2 3    duh
# shape: o d s    oval, diamond, squiggle
# color: r g p    red, green, purple
# fill : e h f    empty, half, full
labels0 = ['1sgh', '3ogf', '3drh', '1orh', '2spe', '1dge', '3sge', '2dpe', '1ogh',
           '1oge', '2sge', '3dgf', '2spf', '2drf', '2orf', '3orf', '3spe', '3sgf',
           '3sph', '2orh', '3dpe', '1dgh', '2opf', '3orh', '3dph', '2ogf', '1sgf',
           '2oph', '1drf', '2sph', '3spf', '2sre', '2sgh', '1sph', '3ore', '2srf',
           '2dgf', '2dgh', '3ope', '2ore', '2ope', '3srh', '3drf', '1ope', '1ogf',
           '2dph', '1dgf', '2dge', '3dgh', '3opf', '2dpf', '2sgf', '1dpe', '3ogh',
           '1sre', '3sgh', '1dpf', '1ore', '3sre', '1sge', '3dre', '1dre', '2dre',
           '2ogh', '1spe', '1dph', '1opf', '2srh', '1oph', '1srf', '1srh', '2drh',
           '3oph', '2oge', '1drh', '3srf', '1orf', '3dge', '3dpf', '3oge', '1spf']

sortedlabels = sorted(labels0)
'''
for i in range(81):
    maplabel = {'d': 'diamond', 'o': 'oval', 's': 'squiggle',
                'r': 'red', 'g': 'green', 'p': 'purple',
                'e': 'empty', 'h': 'striped', 'f': 'filled'}
    print(sortedlabels[i][0] + " " + maplabel[sortedlabels[i][1]] + " "
          + maplabel[sortedlabels[i][2]] + " " + maplabel[sortedlabels[i][3]])
    print("- " + sortedlabels[i])'''

labels1 = ['1sgh', '3ogf', '3drh', '2spe', '1orh', '1ogh', '2orf', '1dge', '3sgf',
           '2drf', '3sge', '1oge', '2sge', '3orf', '2spf', '2dpe', '3dgf', '3dpe',
           '3dph', '3spe', '2opf', '1drf', '1dgh', '3spf', '3orh', '3sph', '2oph',
           '2ogf', '2sgh', '1sgf', '2orh', '2sph', '3ore', '1sph', '3ope', '2dgf',
           '2ope', '2dgh', '3srh', '2dph', '2ore', '2srf', '1ope', '1ogf', '2sre',
           '2dge', '2sgf', '3dgh', '2dpf', '3sgh', '1sge', '1sre', '1dpe', '3dre',
           '3ogh', '1dre', '1dgf', '1ore', '3drf', '1dpf', '1spe', '1srh', '2srh',
           '3oph', '2ogh', '1opf', '2drh', '3opf', '3sre', '1drh', '1orf', '2oge',
           '3dge', '1oph', '3dpf', '1dph', '2dre', '3oge', '1spf', '3srf', '1srf']

labels2 = ['1dpf', '1oph', '2dpf', '3dgf', '1orf', '1dph', '2oge', '3dgh', '2sgh',
           '2srh', '1drh', '2dph', '2dpe', '3oph', '2dre', '2dgf', '2orh', '2spe',
           '2sre', '2dge', '3ope', '1sph', '3orf', '2oph', '2orf', '1ogf', '1sre',
           '1dgf', '3dre', '2dgh', '3ore', '1ope', '1ogh', '1srh', '2sph', '1drf',
           '3orh', '1dre', '1dpe', '3dpe', '3drf', '3sgh', '3opf', '2spf', '3srf',
           '1spf', '1dgh', '2srf', '3sre', '2sgf', '1spe', '3ogf', '1srf', '3sph',
           '2opf', '3dph', '3sgf', '1ore', '1orh', '3drh', '3srh', '3ogh', '3dpf',
           '1sgf', '2ore', '2ogh', '1dge', '3oge', '1opf', '2ogf', '1oge', '2drf',
           '3sge', '3spe', '3dge', '1sgh', '2ope', '2sge', '2drh', '3spf', '1sge']

labels3 = ['1spf', '2sge', '1orh', '3ore', '1drf', '1sge', '3drh', '1sre', '1ogf',
           '1ogh', '2ope', '3dpf', '1dph', '1dgh', '3sgh', '3srf', '2drf', '2ogf',
           '3dge', '3drf', '3orh', '1sph', '3ogh', '3spe', '1dge', '3ope', '1drh',
           '2dgh', '1dpf', '1oge', '1srh', '2drh', '3spf', '2dph', '2sph', '1oph',
           '3sgf', '2sgh', '2spf', '2orf', '2spe', '2srf', '2dre', '2oph', '3oge',
           '3ogf', '3sge', '1sgh', '3oph', '3dpe', '3sph', '1opf', '2opf', '3dre',
           '1sgf', '1ore', '2oge', '2dgf', '1dpe', '3dph', '2sre', '2ogh', '2sgf',
           '1ope', '1dgf', '3dgh', '1srf', '1spe', '3srh', '2dge', '2srh', '1orf',
           '3opf', '3sre', '3dgf', '2orh', '2ore', '2dpe', '2dpf', '1dre', '3orf']

labels4 = ['1dgh', '2opf', '2ore', '3dgf', '1dgf', '3sgh', '3dre', '1sph', '2spe',
           '1sge', '3orh', '2ogf', '3sre', '3srf', '1dpf', '2oge', '1sre', '3dgh',
           '2dph', '3spf', '3oph', '3spe', '1orf', '2dgf', '3drf', '1drf', '3srh',
           '1sgh', '2spf', '1dge', '2dgh', '3orf', '3dph', '2sge', '2dpf', '2dpe',
           '3ogf', '3ore', '3sge', '1orh', '1dpe', '2srf', '3dpf', '2oph', '2srh',
           '1dre', '3opf', '1srf', '3dge', '2sgf', '1spf', '1dph', '2orf', '2sgh',
           '1ope', '1srh', '2sre', '2ope', '1ogh', '2orh', '3dpe', '3oge', '2drf',
           '3drh', '3ogh', '1opf', '1ogf', '1drh', '1spe', '3sgf', '2ogh', '1ore',
           '1sgf', '2drh', '2sph', '1oge', '3ope', '1oph', '3sph', '2dre', '2dge']

labels5 = ['1ope', '2dpe', '3ogh', '3dph', '1opf', '3dgf', '2dgh', '2sge', '2sph',
           '3sge', '1orf', '1drf', '1orh', '3dge', '2ogh', '3drh', '2ore', '3dpe',
           '2ope', '3spf', '3sre', '3ogf', '3ore', '2srf', '1drh', '2dpf', '1srf',
           '2sgf', '3opf', '2drf', '3drf', '2srh', '3srf', '1dpf', '3spe', '1srh',
           '1dre', '1dgh', '1spf', '2oge', '1sge', '1sre', '3sgh', '1dpe', '1ogh',
           '3oge', '1oph', '2dge', '2sgh', '1dgf', '3orh', '1sgf', '1sph', '3dpf',
           '1dph', '3srh', '3dre', '3ope', '2orf', '3oph', '2dre', '1ore', '3dgh',
           '2sre', '2opf', '2dph', '1spe', '2drh', '2oph', '2spe', '1ogf', '1sgh',
           '2spf', '3sgf', '2dgf', '3orf', '1oge', '2orh', '3sph', '1dge', '2ogf']

labels6 = ['2sge', '3dgf', '1orh', '2dpe', '3ogh', '2sph', '3drh', '1opf', '3dge',
           '1orf', '1ope', '3dph', '3ogf', '3spf', '3sge', '2ogh', '2ope', '2dgh',
           '1srf', '2ore', '2dpf', '1drf', '3sre', '1drh', '3srf', '3dpe', '1dpf',
           '2srf', '3opf', '2srh', '1spf', '3ore', '1sge', '2drf', '3oge', '2oge',
           '1srh', '2sgf', '2sgh', '3dpf', '3spe', '1oph', '1sre', '3drf', '3oph',
           '1dgh', '1ore', '1dph', '2dge', '3dgh', '1dre', '2dre', '1sph', '1ogh',
           '1spe', '2drh', '2dph', '2spf', '3ope', '3sgh', '1sgh', '3orh', '1dpe',
           '2orf', '1sgf', '2spe', '3sgf', '2sre', '1ogf', '2dgf', '1dgf', '3srh',
           '3sph', '1oge', '2oph', '3orf', '2opf', '3dre', '1dge', '2orh', '2ogf']
'''
i = 0
folder = "./cards6"
for image in os.listdir(folder):
    #\if len(image) == 5:
    #   os.rename(folder + "/" + image, folder + "/0" + image)
    os.rename(folder + "/" + image, folder + "/" + labels6[i] + ".png")
    i += 1'''

#random.seed(123456789)
num_backgrounds = 50000
num_decks = 7
'''
p = Augmentor.Pipeline("./backgrounds/")

p.flip_left_right(probability=0.5)
p.flip_top_bottom(probability=0.5)
p.random_brightness(probability=0.15, min_factor=0.7, max_factor=1.2)
p.rotate(probability=0.7, max_left_rotation=20, max_right_rotation=20)
p.skew(probability=0.4, magnitude=0.5)
p.zoom(probability=0.2, min_factor=1.1, max_factor=1.5)
p.shear(probability=0.1, max_shear_right=5, max_shear_left=5)
p.gaussian_distortion(probability=0.3, grid_width=5, grid_height=5, magnitude=5,corner='bell', method='in')
p.sample(50000)'''

'''cardfolder = './cardssorted'

for label in labels1:
    os.makedirs(cardfolder+'/'+label)'''
'''
for i in range(num_decks):
    for image in os.listdir('./cards' + str(i)):
        cardcategory = image.split(".")[0]
        shutil.copyfile('./cards' + str(i) + '/' + image,
                        './cardssorted/' + cardcategory + "/" + str(i) + '.png')'''
samples_img = 10000
# STOLEN FROM STACKEXCHANGE
# https://stackoverflow.com/questions/40895785/using-opencv-
#       to-overlay-transparent-image-onto-another-image
# edits background to include foreground at row, col
def transparentoverlay(foreground, background, row, col):
    # normalize alpha channels from 0-255 to 0-1
    row_1 = row + foreground.shape[0]
    col_1 = col + foreground.shape[1]
    alpha_background = background[row:row_1, col:col_1, 3] / 255.0
    alpha_foreground = foreground[:,:,3] / 255.0

    # set adjusted colors
    for color in range(0, 3):
        background[row:row_1, col:col_1, color] = alpha_foreground * foreground[:, :, color] + \
                                              alpha_background * background[row:row_1, col:col_1, color] * (1 - alpha_foreground)

    # set adjusted alpha and denormalize back to 0-255
    background[row:row_1, col:col_1, 3] = (1 - (1 - alpha_foreground) * (1 - alpha_background)) * 255

allimgboxes = []
for i in range(samples_img): #num_backgrounds for big set
    cards = random.sample(labels0, 4)
    rr = random.randrange(0,2)
    for k in range(4): cards[k] = cards[k][:-1] + 'e' if rr == 0 else cards[k][:-1] + 'h'
    scale = random.randrange(27, 50) / 10
    maxcoord = 416-850/scale
    coor1 = (20, int(maxcoord / 2.5)+20)
    coor2 = (200+int(20/scale), int(200+maxcoord/2.5))
    pos = [(random.randrange(coor1[0], coor1[1]), random.randrange(coor1[0], coor1[1])),
           (random.randrange(coor2[0], coor2[1]), random.randrange(coor2[0], coor2[1])),
           (random.randrange(coor2[0], coor2[1]), random.randrange(coor1[0], coor1[1])),
           (random.randrange(coor1[0], coor1[1]), random.randrange(coor2[0], coor2[1]))]

    transparentsum = np.zeros((600,600,4), dtype=np.uint8)
    boxes = []
    for idx in range(4):
        c = cards[idx]
        row, col = pos[idx]
        num = random.randrange(0, num_decks)
        card = cv.imread('./cardssorted/' + c + "/" + str(num) + '.png', cv.IMREAD_UNCHANGED)
        card = cv.resize(card, (int(card.shape[1]/scale), int(card.shape[0]/scale)))
        card = imutils.rotate_bound(card, random.randrange(-10,10))
        transparentoverlay(card, transparentsum, int(row) + 50, int(col) + 100)
        boxes.append([int(col) + 100, int(row) + 50, int(col) + 100 + card.shape[1], int(row) + 50 + card.shape[0]]) #x1, y1, x2, y2


    bbs = BoundingBoxesOnImage([
        BoundingBox(x1=boxes[0][0], y1=boxes[0][1], x2=boxes[0][2], y2=boxes[0][3], label=cards[0]),
        BoundingBox(x1=boxes[1][0], y1=boxes[1][1], x2=boxes[1][2], y2=boxes[1][3], label=cards[1]),
        BoundingBox(x1=boxes[2][0], y1=boxes[2][1], x2=boxes[2][2], y2=boxes[2][3], label=cards[2]),
        BoundingBox(x1=boxes[3][0], y1=boxes[3][1], x2=boxes[3][2], y2=boxes[3][3], label=cards[3])
    ], shape=cv.cvtColor(transparentsum, cv.COLOR_BGRA2RGB).shape)
    allimgboxes.append(bbs)
    cv.imwrite('./temp/' + str(i) + '.png', transparentsum)

sometimes = lambda aug: iaa.Sometimes(0.75, aug)
seq = iaa.Sequential([
    sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.11), fit_output=True, keep_size=False)),
    sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.11), fit_output=True, keep_size=False)),
    iaa.Rotate((-180, 180))
])
few = lambda aug: iaa.Sometimes(0.25, aug)
seqcolor = iaa.Sequential([
    iaa.Multiply((0.5, 1.5)),
    iaa.LinearContrast((0.8, 1.2)),
    iaa.ChangeColorTemperature((3000, 8000)),
    sometimes(iaa.AdditiveGaussianNoise(scale=(0, 20))),
    few(iaa.GaussianBlur(sigma=(0.0, 3.0)))
])

for i in range(samples_img):
    image = cv.imread('./temp/' + str(i) + '.png', cv.IMREAD_UNCHANGED)
    bbs = allimgboxes[i]
    image_aug, allimgboxes[i] = seq(image=image, bounding_boxes=bbs)
    alpha = image_aug[:,:,3]
    image_aug = seqcolor(image=image_aug[:, :, :3])
    image_aug = np.dstack((image_aug, alpha))
    cv.imwrite('./temp/' + str(i) + '.png', image_aug)

'''p = Augmentor.Pipeline("./temp")
p.skew(probability=1, magnitude=0.3)
p.rotate(probability=0.7, max_left_rotation=25, max_right_rotation=25)
p.sample(samples_img)'''

for i in range(samples_img):
    transparentsum = cv.imread("./temp/" + str(i) + ".png", cv.IMREAD_UNCHANGED)
    bot = np.max(np.nonzero(transparentsum)[0])
    right = np.max(np.nonzero(transparentsum)[1])
    top = np.min(np.nonzero(transparentsum)[0])
    left = np.min(np.nonzero(transparentsum)[1])
    transparentsum = transparentsum[top:bot, left:right]
    maxside = max(bot-top, right-left)
    emptysquare = np.zeros((maxside+100, maxside+100, 4), dtype=np.uint8)

    if bot - top > right - left:
        row = random.randrange(0, 80)
        col = random.randrange(0, 80 + int(((bot - top) - (right - left))/2))
    else:
        row = random.randrange(0, 80 + int(((right - left) - (bot - top))/2))
        col = random.randrange(0, 80)
    transparentoverlay(transparentsum, emptysquare, int(row), int(col))
    for idx in range(4):
        allimgboxes[i][idx].x1 += -left + col
        allimgboxes[i][idx].x2 += -left + col
        allimgboxes[i][idx].y1 += -top + row
        allimgboxes[i][idx].y2 += -top + row

        allimgboxes[i][idx].x1 *= 416 / emptysquare.shape[0]
        allimgboxes[i][idx].x2 *= 416 / emptysquare.shape[0]
        allimgboxes[i][idx].y1 *= 416 / emptysquare.shape[0]
        allimgboxes[i][idx].y2 *= 416 / emptysquare.shape[0]
    emptysquare = cv.resize(emptysquare, (416,416))
    #ia.imshow(allimgboxes[i].draw_on_image(cv.cvtColor(emptysquare, cv.COLOR_BGRA2RGB), size=2))
    cv.imwrite("./resized/" + str(i) + ".png", emptysquare)
    i+=1

print('hi')
allbkgnds = os.listdir('./backgroundsaug')
for i in range(samples_img):
    b = random.sample(allbkgnds, 1)[0]
    back = cv.imread('./backgroundsaug/' + b)
    back = cv.cvtColor(back, cv.COLOR_BGR2BGRA)
    cards = cv.imread('./resized/' + str(i) + ".png", cv.IMREAD_UNCHANGED)
    transparentoverlay(cards, back, 0, 0)
    back = cv.cvtColor(back, cv.COLOR_BGRA2BGR)
    cv.imwrite('./output/' + str(i) + ".jpg", back)
    with open('./output/' + str(i) + '.txt', 'w') as f:
        for idx in range(4):
            f.write(str(sortedlabels.index(allimgboxes[i][idx].label)) + " " +
                    str(allimgboxes[i][idx].center_x/416) + " " + str(allimgboxes[i][idx].center_y/416) + " " +
                    str(allimgboxes[i][idx].width/416) + " " + str(allimgboxes[i][idx].height/416))
            if idx < 3: f.write('\n')
