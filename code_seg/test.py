
import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from PIL import Image
import skimage.morphology as morph
import scipy.ndimage.morphology as ndi_morph
from skimage import measure, io

from model import create_model
import utils
from accuracy import compute_metrics

from options import Options
from my_transforms import get_transforms


def main(opt):
    opt.isTrain = False

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.test['gpus'])

    img_dir = opt.test['img_dir']
    label_dir = opt.test['label_dir']
    save_dir = opt.test['save_dir']
    model_path = opt.test['model_path']
    save_flag = opt.test['save_flag']
    if save_flag and not os.path.exists(save_dir):
        os.mkdir(save_dir)

    opt.save_options()

    # data transforms
    test_transform = get_transforms(opt.transform['test'])

    model = create_model(opt.model['name'], opt.model['out_c'], opt.model['pretrained'])
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    cudnn.benchmark = True

    # ----- load trained model ----- #
    print("=> loading trained model")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded model at epoch {}".format(checkpoint['epoch']))
    model = model.module

    # switch to evaluate mode
    model.eval()
    counter = 0
    print("=> Test begins:")

    img_names = os.listdir(img_dir)

    if save_flag:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        strs = img_dir.split('/')
        prob_maps_folder = '{:s}/{:s}_prob_maps'.format(save_dir, strs[-1])
        seg_folder = '{:s}/{:s}_segmentation'.format(save_dir, strs[-1])
        if not os.path.exists(prob_maps_folder):
            os.mkdir(prob_maps_folder)
        if not os.path.exists(seg_folder):
            os.mkdir(seg_folder)

    metric_names = ['acc', 'p_F1', 'dice', 'aji']
    test_results = dict()
    all_result = utils.AverageMeter(len(metric_names))

    for img_name in img_names:
        # load test image
        # print('=> Processing image {:s}'.format(img_name))
        img_path = '{:s}/{:s}'.format(img_dir, img_name)
        img = Image.open(img_path)
        ori_h = img.size[1]
        ori_w = img.size[0]
        name = os.path.splitext(img_name)[0]
        label_path = '{:s}/{:s}_label.png'.format(label_dir, name)
        gt = io.imread(label_path)
     
        input = test_transform((img,))[0].unsqueeze(0)

        # print('\tComputing output probability maps...')
        prob_maps = utils.get_probmaps(input, model, opt)
        pred = np.argmax(prob_maps, axis=0)  # prediction

        pred_labeled = measure.label(pred)
        pred_labeled = morph.remove_small_objects(pred_labeled, opt.post['min_area'])
        pred_labeled = ndi_morph.binary_fill_holes(pred_labeled > 0)
        pred_labeled = measure.label(pred_labeled)

        # print('\tComputing metrics...')
        metrics = compute_metrics(pred_labeled, gt, metric_names)

        # save result for each image
        test_results[name] = [metrics['acc'], metrics['p_F1'], metrics['dice'], metrics['aji']]

        # update the average result
        all_result.update([metrics['acc'], metrics['p_F1'], metrics['dice'], metrics['aji']])

        # save image
        if save_flag:
            # print('\tSaving image results...')
            io.imsave('{:s}/{:s}_pred.png'.format(prob_maps_folder, name), (pred_labeled>0).astype(np.uint8) * 255)
            io.imsave('{:s}/{:s}_prob.png'.format(prob_maps_folder, name), prob_maps[1, :, :])
            final_pred = Image.fromarray(pred_labeled.astype(np.uint16))
            final_pred.save('{:s}/{:s}_seg.tiff'.format(seg_folder, name))

            # save colored objects
            pred_colored_instance = np.zeros((ori_h, ori_w, 3))
            for k in range(1, pred_labeled.max() + 1):
                pred_colored_instance[pred_labeled == k, :] = np.array(utils.get_random_color())
            filename = '{:s}/{:s}_seg_colored.png'.format(seg_folder, name)
            io.imsave(filename, pred_colored_instance)

        counter += 1
        if counter % 10 == 0:
            print('\tProcessed {:d} images'.format(counter))

    print('=> Processed all {:d} images'.format(counter))
    print('Average Acc: {r[0]:.4f}\nF1: {r[1]:.4f}\nDice: {r[2]:.4f}\nAJI: {r[3]:.4f}\n'.format(r=all_result.avg))

    header = metric_names
    utils.save_results(header, all_result.avg, test_results, '{:s}/test_results.txt'.format(save_dir))


if __name__ == '__main__':
    opt = Options(isTrain=False)
    opt.parse()
    opt.print_options()
    main(opt)
