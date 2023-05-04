import numpy as np
import cv2 as cv
import colorsys
from itertools import combinations

# a not-so-good program that plays the game set using opencv!
# is not very robust, does not use neural nets and other fancy stuff
# i've never used opencv before and just wanted to write something
# from scratch for funsies (if i have time i'll do the nets)
# for example, it's bad at differentiating between reds and purples
# and gets confused with shadows and glare (which i tried to fix,
# is not completely fixed but better! than initially)
# also gets confused with overlapping cards, or cards cut off at edges

# loads templates from files for each shape (diamond has two orientations)
shapetemplate = dict()
for name in ['diamond1', 'diamond2', 'oval', 'squiggle', 'diamond1empty',
             'diamond2empty', 'ovalempty', 'squiggleempty']:
    template = cv.imread('templates/' + name + '.jpg')
    template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
    shapetemplate[name] = template

# attempts to set white point to background of card, messes up with shadows
def backgroundWhite(im):
    im = im.astype('float32')
    rw, gw, bw = (im[40,-40] + im[40,-40] + im[40,-40] + im[40,-40]) / 4
    im *= np.array([255/rw, 255/gw, 255/bw], np.float32)
    np.clip(im, 0, 255, im)
    return im.astype('uint8')

# given some corners, tries to find the top-left, top-right,
# bottom-left, and bottom right corners
def getCornerPointsRect(rect, vert=True):
    rect = rect.reshape((len(rect), 2))

    coordsum = np.sum(rect, axis=1)
    coorddiff = np.diff(rect, axis=1)

    topleft = rect[np.argmin(coordsum)]
    bottomright = rect[np.argmax(coordsum)]
    topright = rect[np.argmin(coorddiff)]
    bottomleft = rect[np.argmax(coorddiff)]

    if (topright[0] - topleft[0] > bottomleft[1] - topleft[1] and vert) or \
       (topright[0] - topleft[0] < bottomleft[1] - topleft[1] and not vert):
        bottomleft, topleft,  bottomright, topright = \
        topleft,    topright, bottomleft,  bottomright

    rectpts = np.array([topleft, topright, bottomleft, bottomright], np.float32)
    return rectpts

# tries to do some preprocessing on image to get rid of shadows
def removeShadows(im):
    im = cv.medianBlur(im, 5)
    division = cv.divide(im, cv.GaussianBlur(im, (211,211), 0), scale=180)

    im = cv.GaussianBlur(division, (5,5), 0)
    r, threshhold = cv.threshold(im,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    return threshhold

# extracts the cards from the initial image, basically gets contours, and finds the ones
# that are approximately rectangular and then uses perspective warp to straighten card
def findCardsFromImage(im):
    grayscale = removeShadows(cv.cvtColor(im, cv.COLOR_BGR2GRAY))
    blurred = cv.bilateralFilter(grayscale, 10, 100, 100)
    blurred = cv.GaussianBlur(blurred, (5, 5), 0)
    ret, thresh = cv.threshold(blurred, 127, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    filtercontours = []
    for c in contours:
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            filtercontours.append(c)
    contours = filtercontours

    contours = sorted(contours, key=cv.contourArea, reverse=True)
    largest_card_area = cv.contourArea(contours[0])

    contourtoimage = []
    for i in range(len(contours)):
        card = contours[i]
        if cv.contourArea(card) < 0.28 * largest_card_area:
            break
        peri = cv.arcLength(card, True)
        approx = cv.approxPolyDP(card, 0.04 * peri, True)
        rectpts = getCornerPointsRect(approx)

        straight = np.array([[0, 0], [499, 0], [0, 699], [499, 699]], np.float32)
        transform = cv.getPerspectiveTransform(rectpts, straight)
        warp = cv.warpPerspective(im, transform, (500, 700))
        contourtoimage.append((card, warp))

    return contourtoimage

# gets the symbols on each card, tries to find their count, shape, color,
# and fill using various, and varyingly ungood methods
def getTokensFromCard(im):
    grayscale = cv.cvtColor(~im, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(grayscale, (5, 5), 0)
    edges = cv.Canny(blurred, 5, 60)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    dilated = cv.dilate(edges, kernel)
    ret, thresh = cv.threshold(dilated, 200, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if len(contours) == 0: return
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    largest_token_area = cv.contourArea(contours[0])

    count = 0
    shape = ""
    color = ""
    fill = ""
    for i in range(len(contours)):
        token = contours[i]
        if cv.contourArea(token) < 0.8 * largest_token_area:
            break
        count += 1
        # could go through and check all the tokens and compare/contrast, but...
        if count == 1:
            rect = cv.minAreaRect(token)
            box = cv.boxPoints(rect)
            box = np.int32(box)
            box = getCornerPointsRect(box, False)

            straight = np.array([[0, 0], [229, 0], [0, 99], [229, 99]], np.float32)
            transform = cv.getPerspectiveTransform(box, straight)
            warp = cv.warpPerspective(im, transform, (230, 100))

            grayscale = cv.cvtColor(warp, cv.COLOR_BGR2GRAY)
            ret, thresh = cv.threshold(grayscale, 200, 255, cv.THRESH_BINARY)

            mindiff = 999999999 #joy
            bestshape = ""
            for shape in shapetemplate.keys():
                diff = np.sum(cv.absdiff(thresh, shapetemplate[shape]))
                if diff < mindiff:
                    mindiff = diff
                    bestshape = shape
            if 'oval' in bestshape:
                shape = 'oval'
            elif 'squiggle' in bestshape:
                shape = 'squiggle'
            else:
                shape = 'diamond'

            c = cv.mean(warp)[:3]
            h, s, v = colorsys.rgb_to_hsv(c[2]/255, c[1]/255, c[0]/255)
            h *= 180
            if 0 <= h <= 30 or 170 <= h <= 180:
                color = 'red'
            elif 30 <= h <= 90:
                color = 'green'
            else:
                color = 'purple'

            warp = cv.cvtColor(warp, cv.COLOR_BGR2HSV)
            otherfill = bestshape + "empty" if "empty" not in bestshape else bestshape[:-5]
            emptyfill = bestshape if "empty" in bestshape else otherfill
            filled = bestshape if "empty" not in bestshape else otherfill
            shrink = cv.resize(shapetemplate[emptyfill], (216,90), interpolation=cv.INTER_LINEAR)
            shrink = cv.copyMakeBorder(shrink, 5, 5, 7, 7, borderType=cv.BORDER_CONSTANT, value=(255,255,255))
            avg = cv.mean(warp, cv.absdiff(shapetemplate[filled], cv.bitwise_and(shapetemplate[emptyfill], shrink)))
            if 0 <= avg[1] < 25:
                fill = 'empty'
            elif avg[1] < 100:
                fill = 'stripes'
            else:
                fill = 'filled'

    return count, shape, color, fill

# finds a set in the cards (there may be multiple, but am lazy so went with 1)
def findSet(allcards):
    i = {'oval': 1, 'diamond': 2, 'squiggle': 3,
         'red': 1, 'green': 2, 'purple': 3,
         'empty': 1, 'stripes': 2, 'filled': 3}
    for card1, card2, card3 in combinations(allcards, 3):
        if card1 is None or card2 is None or card3 is None: continue
        count1, shape1, color1, fill1 = card1
        count2, shape2, color2, fill2 = card2
        count3, shape3, color3, fill3 = card3
        # is a set when all sums are multiple of 3
        if (count1 + count2 + count3) % 3 == 0 and \
           (i[shape1] + i[shape2] + i[shape3]) % 3 == 0 and \
           (i[color1] + i[color2] + i[color3]) % 3 == 0 and \
           (i[fill1]  + i[fill2]  + i[fill3] ) % 3 == 0:
            return card1, card2, card3
    else:
        return None

# huzzah!
im = cv.imread('testing/IMG_1059.jpg')
contourtoimage = findCardsFromImage(im)
allcards = []
contourtocard = []
for contour, image in contourtoimage:
    image = backgroundWhite(image)
    card = getTokensFromCard(image)
    allcards.append(card)
    contourtocard.append((contour, card))
match = findSet(allcards)
if match is not None:
    contours = [card[0] for card in contourtocard if card[1] in match]
    cv.drawContours(im, contours, -1, (255, 255, 0), 5)
    cv.imshow('set', im)
    cv.waitKey(0)
    cv.imwrite('good2.jpg', im)
else:
    print('no sets! probably. or i made a whoopsie.')