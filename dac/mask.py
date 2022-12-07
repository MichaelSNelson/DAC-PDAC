import numpy as np
import cv2
import copy
import torch
from dac.utils import save_image, normalize_image, image_to_tensor
from dac.utils import normalize_image, save_image
from dac_networks import run_inference, init_network
import torch.nn.functional as F

def get_mask(attribution, real_img, fake_img, real_class, fake_class, 
             net_module, checkpoint_path, input_shape, input_nc, output_classes,
             downsample_factors=None, sigma=11, struc=10):
    """
    attribution: 2D array <= 1 indicating pixel importance
    """

    net = init_network(checkpoint_path, input_shape, net_module, input_nc, eval_net=True, require_grad=False, output_classes=output_classes,
                       downsample_factors=downsample_factors)
    result_dict = {}
    img_names = ["attr", "real", "fake", "hybrid", "mask_real", "mask_fake", "mask_residual", "mask_weight"]
    imgs_all = []

    a_min = -1
    a_max = 1
    steps = 200
    a_range = a_max - a_min
    step = a_range/float(steps)
    for k in range(0,steps+1):
        thr = a_min + k * step
        copyfrom = copy.deepcopy(real_img)
        copyto = copy.deepcopy(fake_img)
        copyto_ref = copy.deepcopy(fake_img)
        copied_canvas = np.zeros(np.shape(copyfrom))
        mask = np.array(attribution > thr, dtype=np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(struc,struc))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask_size = np.sum(mask)
        mask_cp = copy.deepcopy(mask)

        mask_weight = cv2.GaussianBlur(mask_cp.astype(np.float), (sigma,sigma),0)
        #print(copyto.shape)
        #print(copyfrom.shape)
        #print(mask_weight.shape)
        mask_weight = np.repeat(mask_weight[:,:, np.newaxis], 3, axis=2)
        copyto = np.array((copyto * (1 - mask_weight)) + (copyfrom * mask_weight), dtype=np.float)

        copied_canvas += np.array(mask_weight*copyfrom)
        copied_canvas_to = np.zeros(np.shape(copyfrom))
        copied_canvas_to += np.array(mask_weight*copyto_ref)
        diff_copied = copied_canvas - copied_canvas_to
        
        fake_img_norm = normalize_image(copy.deepcopy(fake_img))
        fake_img_norm = torch.transpose(torch.squeeze(image_to_tensor(normalize_image(fake_img_norm).astype(np.float32))), 2,0).unsqueeze(0)
        #print(fake_img_norm.shape)
        out_fake =  F.softmax(net(fake_img_norm), dim=1)
        
        real_img_norm = normalize_image(copy.deepcopy(real_img))
        real_img_norm = torch.transpose(torch.squeeze(image_to_tensor(normalize_image(real_img_norm).astype(np.float32))), 2,0).unsqueeze(0)
        #print(fake_img_norm.shape)
        out_real =  F.softmax(net(real_img_norm), dim=1)

        im_copied_norm = normalize_image(copy.deepcopy(copyto))
        im_copied_norm = torch.transpose(torch.squeeze(image_to_tensor(normalize_image(im_copied_norm).astype(np.float32))), 2,0).unsqueeze(0)
        #print(fake_img_norm.shape)
        out_copyto =  F.softmax(net(im_copied_norm), dim=1)        


        imgs = [attribution, real_img_norm, fake_img_norm, im_copied_norm, normalize_image(copied_canvas), 
                normalize_image(copied_canvas_to), normalize_image(diff_copied), mask_weight]

        imgs_all.append(imgs)

        mrf_score = out_copyto[0][real_class] - out_fake[0][real_class]     
        result_dict[thr] = [float(mrf_score.detach().cpu().numpy()), mask_size]

    return result_dict, img_names, imgs_all
