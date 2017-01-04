import sys
sys.path.append( '../tools' )
import FFDio
import mxnet as mx
import numpy as np
import os
import cv2

norm_width = 224
norm_height = 224

def predict(model, img, gt):
    xs = gt[0, :]
    ys = gt[1, :]

    x1 = min(xs)
    y1 = min(ys)
    x2 = max(xs)
    y2 = max(ys)
    w = x2 - x1
    h = y2 - y1

    avg_wh = (w + h) / 2 / 2
    res = avg_wh * 0.5

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

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

    return pts

def compute_rmse(gt, pts):
    iod = np.linalg.norm( gt[:, 36] - gt[:, 45] )

    sum = 0
    for i in range(0, 68):
        sum += np.linalg.norm( gt[:, i] - pts[:, i] )
    rmse_68 = sum / (68 * iod)

    sum = 0
    for i in range(17, 68):
        sum += np.linalg.norm( gt[:, i] - pts[:, i] )
    rmse_51 = sum / (51 * iod)

    return rmse_68, rmse_51

if __name__ == '__main__':
    # load model
    prefix = 'vgg_16_reduced'
    model = mx.model.FeedForward.load(prefix, 50)

    image_list_indoor = FFDio.collect_data_set('../original_images/300W/01_Indoor')
    image_list_outdoor = FFDio.collect_data_set('../original_images/300W/02_Outdoor')
    image_list_val = image_list_indoor + image_list_outdoor

    errors_68 = []
    errors_51 = []
    for image_name in image_list_val:
        print image_name
        img = cv2.imread(image_name)

        gt_name = image_name[0:image_name.rfind('.')] + '.pts'
        gt = FFDio.load_gt(gt_name)

        predicted_pts = predict(model, img, gt)

        rmse_68, rmse_51 = compute_rmse(gt, predicted_pts)
        #print rmse_68, rmse_51
        errors_68.append(rmse_68)
        errors_51.append(rmse_51)

        '''
        # draw results
        oimg = img.copy()

        for j in range(0, 68):
            cv2.circle(oimg, (int(predicted_pts[0, j]), int(predicted_pts[1, j])), 2, (255, 0, 0), 2)

        cv2.imshow('img', oimg)
        cv2.waitKey(-1)
        '''
    
    sum = 0
    for e in errors_68:
        sum += e
    print 'rmse 68 - ', sum / len(image_list_val)

    sum = 0
    for e in errors_51:
        sum += e
    print 'rmse 51 - ', sum / len(image_list_val)