import sys
sys.path.append( '../tools' )
import FFDio
from FFDIter import FFDIter
import mxnet as mx
import numpy as np
import os
import cv2


if __name__ == '__main__':
    # load CNN model
    prefix = 'vgg_16_reduced'
    model = mx.model.FeedForward.load(prefix, 50)

    # load face detection model
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    norm_width = 224
    norm_height = 224

    batch_size = 30

    image_list_indoor = FFDio.collect_data_set('../original_images/300W/01_Indoor')
    image_list_outdoor = FFDio.collect_data_set('../original_images/300W/02_Outdoor')
    image_list_val = image_list_indoor + image_list_outdoor

    for image_name in image_list_val:
        img = cv2.imread(image_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # detect face
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            avg_wh = (w + h) / 2 / 2
            res = avg_wh * 0.2
            center_x = x + w / 2
            center_y = y + h / 2
            x1 = center_x - avg_wh - res
            x2 = center_x + avg_wh + res
            y1 = center_y - avg_wh - res
            y2 = center_y + avg_wh + res

            src_points = np.float32([[x1, y1], [x1, y2], [x2, y1]])
            dst_points = np.float32([[0, 0], [0, norm_height], [norm_width, 0]])

            warp_mat = cv2.getAffineTransform(src_points, dst_points)

            # warp image
            norm_img = cv2.warpAffine(img, warp_mat, (norm_width, norm_height))

            # swap channel
            data = cv2.cvtColor(norm_img, cv2.COLOR_BGR2RGB)

            # normalize
            data = data.astype(np.float32)
            data -= [128, 128, 128]
            data = data.transpose((2, 0, 1))

            # predict
            temp = mx.nd.array(np.array([data]))
            pts = model.predict(temp)
            pts = pts[0]

            inv_warp_mat = cv2.invertAffineTransform(warp_mat)
            pts = np.reshape(pts, (2, 68))  # 2x68
            pts[0, :] *= norm_img.shape[1]
            pts[1, :] *= norm_img.shape[0]

            # transform predicted shape to original image
            homography_pts = np.concatenate(((pts), np.ones((1, pts.shape[1]))), axis=0)
            pts = np.matmul(inv_warp_mat, homography_pts)

            # draw results
            oimg = img.copy()
            
            cv2.rectangle(oimg,(x,y),(x+w,y+h),(255,0,0),2)

            for j in range(0, 68):
                cv2.circle(oimg, (int(pts[0, j]), int(pts[1, j])), 2, (255, 0, 0), 2)

            cv2.imshow('img', oimg)
            cv2.waitKey(-1)