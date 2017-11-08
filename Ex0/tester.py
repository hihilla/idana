import numpy as np
import cv2
import ex0
import matplotlib.pyplot as plt

def test_2(imageName):
    print(imageName)
    img = cv2.imread(imageName)
    # reorder from BGR to RGB
    img = img[:, :, [2, 1, 0]]

    g_img_Lightness = ex0.convertRGB2Gray(img, 'Lightness')
    g_img_Average = ex0.convertRGB2Gray(img, 'Average')
    g_img_Luminosity = ex0.convertRGB2Gray(img)

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
    print(type(ax1))
    ax1.imshow(img), ax1.set_title('RGB image')
    ax2.imshow(g_img_Lightness, cmap='gray'), ax2.set_title('Grayscale image: Lightness')
    ax3.imshow(g_img_Average, cmap='gray'), ax4.set_title('Grayscale image: Average')
    ax4.imshow(g_img_Luminosity, cmap='gray'), ax4.set_title('Grayscale image: Luminosity')


def test_3(imageName):
    img = cv2.imread(imageName);

    # reorder from BGR to RGB
    img = img[:, :, [2, 1, 0]]

    # normalize to range [0 1]
    img = np.float32(img)
    img = img / 255

    img_yiq = ex0.rgb2yiq(img)
    img_rgb = ex0.yiq2rgb(img_yiq)

    f, ((ax1, ax2)) = plt.subplots(1, 2, sharex='col')
    ax1.imshow(img), ax1.set_title('Original RGB image')
    ax2.imshow(img_rgb), ax2.set_title('rgb2yiq->yiq2rgb image')

    # test 3.
# imageName = './Images/im1.jpg'
# test_3(imageName)
#
# imageName = './Images/peppers.png'
# test_3(imageName)
# test_1_a()


imageName = './Images/im1.jpg'
test_2(imageName)
