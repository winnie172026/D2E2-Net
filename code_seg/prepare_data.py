"""
This script is used to create Voronoi labels and cluster labels from the point labels
and prepare the dataset for training and testing.

Author: Hui Qu
"""


import os
import shutil
import numpy as np
from skimage import morphology, measure, io
from sklearn.cluster import KMeans
from scipy.ndimage.morphology import distance_transform_edt as dist_tranform
import glob
import json

from options import Options


def main(opt):
    # opt.ratio = 0.05
    dataset = opt.dataset
    ratio = opt.ratio
    data_dir = '../data/{:s}'.format(dataset)
    img_dir = '../data/{:s}/images'.format(dataset)
    # label_instance_dir = '../data/{:s}/labels_instance'.format(dataset)
    if opt.ratio == 1:
        label_point_dir = '../data/{:s}/labels_point'.format(dataset)
    else:
        label_point_dir_partial = '../data/{:s}/labels_point_{:.2f}'.format(dataset, ratio)
        label_point_dir = '../data/{:s}/labels_point_{:.2f}_pred'.format(dataset, ratio)
    label_vor_dir = '../data/{:s}/labels_voronoi_{:.2f}'.format(dataset, ratio)
    label_cluster_dir = '../data/{:s}/labels_cluster_{:.2f}'.format(dataset, ratio)
    patch_folder = '../data/{:s}/patches'.format(dataset)
    train_data_dir = '../data_for_train/{:s}'.format(dataset)

    with open('{:s}/train_val_test.json'.format(data_dir), 'r') as file:
        data_list = json.load(file)
        train_list = data_list['train']

    # ------ create point label from detection results
    # create_point_label_from_instance(label_instance_dir, label_point_dir, train_list)
    if opt.ratio < 1:
        if not opt.detection_results_dir:
            opt.detection_results_dir = '../experiments/detection/{:s}/{:.2f}/3/best/images_prob_maps'.format(dataset, ratio)
        create_point_label_from_detection_results(opt.detection_results_dir, label_point_dir_partial, label_point_dir, train_list,
                                                  dist_thresh=opt.radius)

    # ------ create Voronoi label from point label
    create_Voronoi_label(label_point_dir, label_vor_dir, train_list)

    # ------ create cluster label from point label and image
    create_cluster_label(img_dir, label_point_dir, label_vor_dir, label_cluster_dir, train_list)

    # ------ split large images into 250x250 patches
    if os.path.exists(patch_folder):
        shutil.rmtree(patch_folder)
    create_folder(patch_folder)
    print("Spliting large images into small patches...")
    split_patches(img_dir, '{:s}/images'.format(patch_folder))
    split_patches(label_vor_dir, '{:s}/labels_voronoi'.format(patch_folder), 'label_vor')
    split_patches(label_cluster_dir, '{:s}/labels_cluster'.format(patch_folder), 'label_cluster')

    # ------ divide dataset into train, val and test sets
    organize_data_for_training(data_dir, train_data_dir)

    # ------ compute mean and std
    # compute_mean_std(data_dir, train_data_dir)


def create_point_label_from_instance(data_dir, save_dir, train_list):
    def get_point(img):
        a = np.where(img != 0)
        rmin, rmax, cmin, cmax = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
        return (rmin + rmax) // 2, (cmin + cmax) // 2

    create_folder(save_dir)
    print("Generating point label from instance label...")
    image_list = os.listdir(data_dir)
    N_total = len(train_list)
    N_processed = 0
    for image_name in image_list:
        name = image_name.split('.')[0]
        if '{:s}.png'.format(name[:-6]) not in train_list or name[-5:] != 'label':
            continue

        N_processed += 1
        flag = '' if N_processed < N_total else '\n'
        print('\r\t{:d}/{:d}'.format(N_processed, N_total), end=flag)

        image_path = os.path.join(data_dir, image_name)
        image = io.imread(image_path)
        h, w = image.shape

        # extract bbox
        id_max = np.max(image)
        label_point = np.zeros((h, w), dtype=np.uint8)

        for i in range(1, id_max + 1):
            nucleus = image == i
            if np.sum(nucleus) == 0:
                continue
            x, y = get_point(nucleus)
            label_point[x, y] = 255

        io.imsave('{:s}/{:s}_point.png'.format(save_dir, name), label_point.astype(np.uint8))


def create_point_label_from_detection_results(results_dir, label_point_dir_partial, label_point_dir,
                                              train_list, dist_thresh):
    create_folder(label_point_dir)
    print("Generating point label from detection results...")
    N_total = len(train_list)
    N_processed = 0
    for image_name in train_list:
        name = image_name.split('.')[0]

        N_processed += 1
        flag = '' if N_processed < N_total else '\n'
        print('\r\t{:d}/{:d}'.format(N_processed, N_total), end=flag)

        points_partial = io.imread('{:s}/{:s}_label_point.png'.format(label_point_dir_partial, name))
        # label_point = np.zeros(points_pred.shape, dtype=np.uint8)
        label_point = points_partial
        label_point_dilated = morphology.dilation(label_point, morphology.disk(dist_thresh))

        points_pred = io.imread('{:s}/{:s}_pred.png'.format(results_dir, name))   # extract bbox
        points_pred_labeled = measure.label(points_pred)
        point_regions = measure.regionprops(points_pred_labeled)

        new_label_point = label_point.copy()
        for region in point_regions:
            x, y = int(region.centroid[0]), int(region.centroid[1])
            if label_point_dilated[x, y] > 0:
                continue
            else:
                new_label_point[x, y] = 255

        io.imsave('{:s}/{:s}_label_point.png'.format(label_point_dir, name), new_label_point.astype(np.uint8))


def create_Voronoi_label(data_dir, save_dir, train_list):
    from scipy.spatial import Voronoi
    from shapely.geometry import Polygon
    from utils import voronoi_finite_polygons_2d, poly2mask

    create_folder(save_dir)
    print("Generating Voronoi label from point label...")
    N_total = len(train_list)
    N_processed = 0
    for img_name in train_list:
        name = img_name.split('.')[0]
        
        N_processed += 1
        flag = '' if N_processed < N_total else '\n'
        print('\r\t{:d}/{:d}'.format(N_processed, N_total), end=flag)

        img_path = '{:s}/{:s}_label_point.png'.format(data_dir, name)
#         print('path:', img_path)
        label_point = io.imread(img_path)
#         print('label:', label_point.shape)
        h, w = label_point.shape

        points = np.argwhere(label_point > 0)
#         print('points:', len(points))
        
        if len(points) > 4:
            vor = Voronoi(points)
        else:
            continue;

        regions, vertices = voronoi_finite_polygons_2d(vor)
        box = Polygon([[0, 0], [0, w], [h, w], [h, 0]])
        region_masks = np.zeros((h, w), dtype=np.int16)
        edges = np.zeros((h, w), dtype=np.bool)
        count = 1
        for region in regions:
            polygon = vertices[region]
            # Clipping polygon
            poly = Polygon(polygon)
            poly = poly.intersection(box)
            polygon = np.array([list(p) for p in poly.exterior.coords])
#             print(polygon.shape)
           
            mask = poly2mask(polygon[:, 0], polygon[:, 1], (h, w))
            edge = mask * (~morphology.erosion(mask, morphology.disk(1)))
            edges += edge
            region_masks[mask] = count
            count += 1

        # fuse Voronoi edge and dilated points
        label_point_dilated = morphology.dilation(label_point, morphology.disk(2))
        label_vor = np.zeros((h, w, 3), dtype=np.uint8)
        label_vor[:, :, 0] = morphology.closing(edges > 0, morphology.disk(1)).astype(np.uint8) * 255
        label_vor[:, :, 1] = (label_point_dilated > 0).astype(np.uint8) * 255

        io.imsave('{:s}/{:s}_label_vor.png'.format(save_dir, name), label_vor)


def create_cluster_label(data_dir, label_point_dir, label_vor_dir, save_dir, train_list):
    from scipy.ndimage import morphology as ndi_morph

    create_folder(save_dir)
    print("Generating cluster label from point label...")
    N_total = len(train_list)
    N_processed = 0
    for img_name in train_list:
        name = img_name.split('.')[0]
        N_processed += 1
        flag = '' if N_processed < N_total else '\n'
        print('\r\t{:d}/{:d}'.format(N_processed, N_total), end=flag)

        # print('\t[{:d}/{:d}] Processing image {:s} ...'.format(count, len(img_list), img_name))
        ori_image = io.imread('{:s}/{:s}.png'.format(data_dir, name))
        h, w, _ = ori_image.shape
        label_point = io.imread('{:s}/{:s}_label_point.png'.format(label_point_dir, name))

        # k-means clustering
        dist_embeddings = dist_tranform(255 - label_point).reshape(-1, 1)
        clip_dist_embeddings = np.clip(dist_embeddings, a_min=0, a_max=20)
        color_embeddings = np.array(ori_image, dtype=np.float).reshape(-1, 3) / 10
        embeddings = np.concatenate((color_embeddings, clip_dist_embeddings), axis=1)

        # print("\t\tPerforming k-means clustering...")
        kmeans = KMeans(n_clusters=3, random_state=0).fit(embeddings)
        clusters = np.reshape(kmeans.labels_, (h, w))

        # get nuclei and background clusters
        overlap_nums = [np.sum((clusters == i) * label_point) for i in range(3)]
        nuclei_idx = np.argmax(overlap_nums)
        remain_indices = np.delete(np.arange(3), nuclei_idx)
        dilated_label_point = morphology.binary_dilation(label_point, morphology.disk(5))
        overlap_nums = [np.sum((clusters == i) * dilated_label_point) for i in remain_indices]
        background_idx = remain_indices[np.argmin(overlap_nums)]

        nuclei_cluster = clusters == nuclei_idx
        background_cluster = clusters == background_idx

        # refine clustering results
        # print("\t\tRefining clustering results...")
        nuclei_labeled = measure.label(nuclei_cluster)
        initial_nuclei = morphology.remove_small_objects(nuclei_labeled, 30)
        refined_nuclei = np.zeros(initial_nuclei.shape, dtype=np.bool)

        label_vor = io.imread('{:s}/{:s}_label_vor.png'.format(label_vor_dir, img_name[:-4]))
        voronoi_cells = measure.label(label_vor[:, :, 0] == 0)
        voronoi_cells = morphology.dilation(voronoi_cells, morphology.disk(2))

        # refine clustering results
        unique_vals = np.unique(voronoi_cells)
        cell_indices = unique_vals[unique_vals != 0]
        N = len(cell_indices)
        for i in range(N):
            cell_i = voronoi_cells == cell_indices[i]
            nucleus_i = cell_i * initial_nuclei

            nucleus_i_dilated = morphology.binary_dilation(nucleus_i, morphology.disk(5))
            nucleus_i_dilated_filled = ndi_morph.binary_fill_holes(nucleus_i_dilated)
            nucleus_i_final = morphology.binary_erosion(nucleus_i_dilated_filled, morphology.disk(7))
            refined_nuclei += nucleus_i_final > 0

        refined_label = np.zeros((h, w, 3), dtype=np.uint8)
        label_point_dilated = morphology.dilation(label_point, morphology.disk(10))
        refined_label[:, :, 0] = (background_cluster * (refined_nuclei == 0) * (label_point_dilated == 0)).astype(np.uint8) * 255
        refined_label[:, :, 1] = refined_nuclei.astype(np.uint8) * 255

        io.imsave('{:s}/{:s}_label_cluster.png'.format(save_dir, name), refined_label)


def split_patches(data_dir, save_dir, post_fix=None):
    import math
    """ split large image into small patches """
    create_folder(save_dir)

    image_list = os.listdir(data_dir)
    for image_name in image_list:
        name = image_name.split('.')[0]
        if post_fix and name[-len(post_fix):] != post_fix:
            continue
        image_path = os.path.join(data_dir, image_name)
        image = io.imread(image_path)
        seg_imgs = []

        # split into 16 patches of size 250x250
        h, w = image.shape[0], image.shape[1]
        patch_size = 250
        h_overlap = math.ceil((4 * patch_size - h) / 3)
        w_overlap = math.ceil((4 * patch_size - w) / 3)
        for i in range(0, h-patch_size+1, patch_size-h_overlap):
            for j in range(0, w-patch_size+1, patch_size-w_overlap):
                if len(image.shape) == 3:
                    patch = image[i:i+patch_size, j:j+patch_size, :]
                else:
                    patch = image[i:i + patch_size, j:j + patch_size]
                seg_imgs.append(patch)

        for k in range(len(seg_imgs)):
            if post_fix:
                io.imsave('{:s}/{:s}_{:d}_{:s}.png'.format(save_dir, name[:-len(post_fix)-1], k, post_fix), seg_imgs[k])
            else:
                io.imsave('{:s}/{:s}_{:d}.png'.format(save_dir, name, k), seg_imgs[k])


def organize_data_for_training(data_dir, train_data_dir):
    # --- Step 1: create folders --- #
    create_folder(train_data_dir)
    create_folder('{:s}/images/train'.format(train_data_dir))
    create_folder('{:s}/images/val'.format(train_data_dir))
    create_folder('{:s}/images/test'.format(train_data_dir))
    if os.path.exists('{:s}/labels_voronoi'.format(train_data_dir)):
        shutil.rmtree('{:s}/labels_voronoi'.format(train_data_dir))
    if os.path.exists('{:s}/labels_cluster'.format(train_data_dir)):
        shutil.rmtree('{:s}/labels_cluster'.format(train_data_dir))
    create_folder('{:s}/labels_voronoi/train'.format(train_data_dir))
    create_folder('{:s}/labels_cluster/train'.format(train_data_dir))

    # --- Step 2: move images and labels to each folder --- #
    print('Organizing data for training...')
    with open('{:s}/train_val_test.json'.format(data_dir), 'r') as file:
        data_list = json.load(file)
        train_list, val_list, test_list = data_list['train'], data_list['val'], data_list['test']

    # train
    for img_name in train_list:
        name = img_name.split('.')[0]
        # images
        for file in glob.glob('{:s}/patches/images/{:s}*'.format(data_dir, name)):
            file_name = file.split('/')[-1]
            dst = '{:s}/images/train/{:s}'.format(train_data_dir, file_name)
            shutil.copyfile(file, dst)
        # label_voronoi
        for file in glob.glob('{:s}/patches/labels_voronoi/{:s}*'.format(data_dir, name)):
            file_name = file.split('/')[-1]
            dst = '{:s}/labels_voronoi/train/{:s}'.format(train_data_dir, file_name)
            shutil.copyfile(file, dst)
        # label_cluster
        for file in glob.glob('{:s}/patches/labels_cluster/{:s}*'.format(data_dir, name)):
            file_name = file.split('/')[-1]
            dst = '{:s}/labels_cluster/train/{:s}'.format(train_data_dir, file_name)
            shutil.copyfile(file, dst)
    # val
    for img_name in val_list:
        name = img_name.split('.')[0]
        # images
        for file in glob.glob('{:s}/images/{:s}*'.format(data_dir, name)):
            file_name = file.split('/')[-1]
            dst = '{:s}/images/val/{:s}'.format(train_data_dir, file_name)
            shutil.copyfile(file, dst)
    # test
    for img_name in test_list:
        name = img_name.split('.')[0]
        # images
        for file in glob.glob('{:s}/images/{:s}*'.format(data_dir, name)):
            file_name = file.split('/')[-1]
            dst = '{:s}/images/test/{:s}'.format(train_data_dir, file_name)
            shutil.copyfile(file, dst)


def compute_mean_std(data_dir, train_data_dir):
    """ compute mean and standarad deviation of training images """
    total_sum = np.zeros(3)  # total sum of all pixel values in each channel
    total_square_sum = np.zeros(3)
    num_pixel = 0  # total num of all pixels

    with open('{:s}/train_val_test.json'.format(data_dir), 'r') as file:
        data_list = json.load(file)
        train_list = data_list['train']

    print('Computing the mean and standard deviation of training data...')

    for file_name in train_list:
        img_name = '{:s}/images/{:s}'.format(data_dir, file_name)
        img = io.imread(img_name)
        if len(img.shape) != 3 or img.shape[2] < 3:
            continue
        img = img[:, :, :3].astype(int)
        total_sum += img.sum(axis=(0, 1))
        total_square_sum += (img ** 2).sum(axis=(0, 1))
        num_pixel += img.shape[0] * img.shape[1]

    # compute the mean values of each channel
    mean_values = total_sum / num_pixel

    # compute the standard deviation
    std_values = np.sqrt(total_square_sum / num_pixel - mean_values ** 2)

    # normalization
    mean_values = mean_values / 255
    std_values = std_values / 255

    np.save('{:s}/mean_std.npy'.format(train_data_dir), np.array([mean_values, std_values]))
    np.savetxt('{:s}/mean_std.txt'.format(train_data_dir), np.array([mean_values, std_values]), '%.4f', '\t')
    # np.save('{:s}/mean_std.npy'.format(train_data_dir), np.array([mean_values, std_values]))


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


if __name__ == '__main__':
    opt = Options(isTrain=True)
    opt.parse()
    main(opt)
