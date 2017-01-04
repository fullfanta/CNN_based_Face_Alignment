import numpy as np
import random
import cv2
import copy
import glob
import os

def load_gt(gt_name):
    f = open(gt_name, 'rt')


    contents = f.readlines()
    contents = contents[3:3+68]
    f.close()

    xs = []
    ys = []
    for line in contents:
        line = line.rstrip('\n\r')
        xy = line.split(' ')
        xs.append(float(xy[0]))
        ys.append(float(xy[1]))

    pts = np.float32([xs, ys])
    return pts

def save_gt(gt_name, gt):
    f = open(gt_name, 'wt')

    f.write('version: 1\n')
    f.write('n_points: 68\n')
    f.write('{\n')

    for i in range(0, 68):
        f.write('%f %f\n' % (gt[0, i], gt[1, i]) )

    f.write('}\n')

    f.close()


def crop(img, pts, bool_aug = True, aug_num = 1, norm_width = 128, norm_height = 128):

    img_height, img_width = img.shape[:2]

    xs = pts[0, :]
    ys = pts[1, :]

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

    src_points[:, 0] -= center_x
    src_points[:, 1] -= center_y

    norm_images = []

    if bool_aug:

        for i in range(0, aug_num):

            # random perturbation
            rand_scale = 1 + random.random() * 0.6 - 0.3                    # 0.7 ~ 1.3
            rand_angle = (random.random() * 40 - 20) / 180 * np.pi      # -20 degree ~ +20 degree
            rand_tx = (random.random() * 0.4 - 0.2) * avg_wh            # -20% ~ +20%
            rand_ty = (random.random() * 0.4 - 0.2) * avg_wh            # -20% ~ +20%
            new_warp_mat = np.float32([[rand_scale * np.cos(rand_angle), rand_scale * np.sin(rand_angle), rand_tx], [-rand_scale * np.sin(rand_angle), rand_scale * np.cos(rand_angle), rand_ty], [0, 0, 1]])

            homography_pts = np.concatenate((np.transpose(src_points), np.ones((1, src_points.shape[0]))), axis=0)
            norm_pts = np.matmul(new_warp_mat, homography_pts)
            new_src_points = np.transpose(norm_pts[0:2, :])

            new_src_points[:, 0] += center_x
            new_src_points[:, 1] += center_y
            new_src_points = new_src_points.astype('float32')

            # compute transformation parameters
            warp_mat = cv2.getAffineTransform(new_src_points, dst_points)

            # warp image
            norm_img = cv2.warpAffine(img, warp_mat, (norm_width, norm_height))

            # transform GT
            homography_pts = np.concatenate((np.float32([xs, ys]), np.ones((1, len(xs)))), axis=0)
            norm_pts = np.matmul(warp_mat, homography_pts)

            # flip
            if random.randrange(0, 2) == 1:
                norm_img = cv2.flip(norm_img, 1)

                flip_index = [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, \
                                    26, 25, 24, 23, 22, 21, 20, 19, 18, 17, \
                                    27, 28, 29, 30, \
                                    35, 34, 33, 32, 31, \
                                    45, 44, 43, 42, 47, 46, \
                                    39, 38, 37, 36, 41, 40, \
                                    54, 53, 52, 51, 50, 49, 48, 59, 58, 57, 56, 55, \
                                    64, 63, 62, 61, 60, 67, 66, 65]

                temp = copy.deepcopy(norm_pts)
                for i in range(0, 68):
                    norm_pts[0, flip_index[i]] = norm_img.shape[1] - temp[0, i]
                    norm_pts[1, flip_index[i]] = temp[1, i]

            norm_images.append( (norm_img, norm_pts) )  

        if aug_num == 1:
            return norm_images[0]
        else:
            return norm_images
    else:
        # warp image
        norm_img = cv2.warpAffine(img, warp_mat, (norm_width, norm_height))

        # transform GT
        homography_pts = np.concatenate((np.float32([xs, ys]), np.ones((1, len(xs)))), axis=0)
        norm_pts = np.matmul(warp_mat, homography_pts)


        return (norm_img, norm_pts)   

def collect_data_set(img_root):

    img_list = []

    image_list_png = glob.glob(img_root + '/*.png')
    image_list_jpg = glob.glob(img_root + '/*.jpg')
    image_list = image_list_png + image_list_jpg

    for image_name in image_list:
        gt_name = image_name[0:image_name.rfind('.')] + '.pts'

        if os.path.exists(gt_name):
            img_list.append(image_name)

    return img_list


if __name__ == '__main__':

    image_list = collect_data_set('../original_images/AFW')

    for image_name in image_list:
        # image
        img_cv = cv2.imread(image_name)

        # gt
        gt_name = image_name[0:image_name.rfind('.')] + '.pts'
        pts = load_gt(gt_name)
        
        (norm_img, norm_pts) = crop(img_cv, pts, True, 1)

        for i in range(0, 68):
            cv2.circle(norm_img, (int(norm_pts[0, i]), int(norm_pts[1, i])), 2, (255, 0, 0), 2)

        cv2.imshow('temp', norm_img)
        cv2.waitKey(-1)