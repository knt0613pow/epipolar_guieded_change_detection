

import cv2

temp = [[712.3796, 337.1613, 906.2020, 428.7586],
        [719.9413, 341.9176, 873.3006, 398.4065],
        [303.5212, 499.8956, 618.8426, 639.5920]]
temp2 = [[1.4794e+02, 2.7875e+02, 7.8918e+02, 3.9604e+02],
        [6.4373e-01, 0.0000e+00, 7.6970e+01, 3.3295e+02]]

 
im = cv2.imread('/home/kimnamtae1230/epipolar/dataset_pseudo/IndeokWon/pair_data/I1.jpg')
im2 = cv2.imread('/home/kimnamtae1230/epipolar/dataset_pseudo/region2/1_37.6789_126.7546_10.0.jpg')
 
for xy in temp:
    xy = [int(xyxy) for xyxy in xy]
    p1 = xy[0], xy[1]
    p2 = xy[2], xy[3]
    cv2.rectangle(im, p1 , p2, (255,0,0), 2)

for xy in temp2:
    xy = [int(xyxy) for xyxy in xy]
    p1 = xy[0], xy[1]
    p2 = xy[2], xy[3]
    cv2.rectangle(im2, p1 , p2, (255,0,0), 2)
 
cv2.imwrite('result.jpg', im)
cv2.imwrite('result2.jpg', im2)
print(temp)