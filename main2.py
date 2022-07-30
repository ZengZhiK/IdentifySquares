import cv2

import imageio

if __name__ == '__main__':
    img_list = []
    for i in range(0, 737):
        img = cv2.imread('./imgs/{}.png'.format(i), cv2.IMREAD_UNCHANGED)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        cv2.imshow('img', img)
        cv2.waitKey(1)
        img_list.append(img)
    imageio.mimsave('./result/video.gif', img_list, fps=60)
