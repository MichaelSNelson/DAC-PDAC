import numpy as np
import cv2
import copy
import torch
from dac.utils import save_image, normalize_image, image_to_tensor
from dac.utils import normalize_image, save_image
from dac_networks import run_inference, init_network
import torch.nn.functional as F
from PIL import Image

def get_mask(attribution, real_img, fake_img, real_class, fake_class, 
             net_module, checkpoint_path, input_shape, input_nc, output_classes,
             downsample_factors=None, sigma=11, struc=10):
    """
    attribution: 2D array <= 1 indicating pixel importance
    """
    #print('first call attribution', str(attribution.shape))
    if attribution.shape[0] == 3:
        attribution = np.transpose(attribution, (1,2,0))
    #attribution = np.mean(attribution, axis=2)
    #print('second call attribution', str(attribution.shape))
    net = init_network(checkpoint_path, input_shape, net_module, input_nc, eval_net=True, require_grad=False, output_classes=output_classes,
                       downsample_factors=downsample_factors)
    result_dict = {}
    img_names = ["attr", "real", "fake", "hybrid", "mask_real", "mask_fake", "mask_residual", "mask_weight"]
    imgs_all = []
    #####figuring out noise this is not fine
    im = Image.fromarray(attribution,'RGB')
    im.save('getMask_attribution.png')
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
        #######figuring out noise
        #im = Image.fromarray(copyto_ref,'RGB')
        #im.save('test_fake_img_copy.png')
        ###########This is fine
        mask = np.array(attribution > thr, dtype=np.uint8)
        #change flattening to after masking
        mask = np.mean(mask, axis=2)
        #print(mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(struc,struc))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask_size = np.sum(mask)
        mask_cp = copy.deepcopy(mask)
        #print("initial maskcp ", str(mask_cp.shape))
        mask_weight = cv2.GaussianBlur(mask_cp.astype(np.float), (sigma,sigma),0)
        #print("copyto", str(copyto.shape))
        #print("copyfrom ", str(copyfrom.shape))
        #print("maskweight ", str(mask_weight.shape))
        #reshape mask due to channel axis mismatch with copyto and copyfrom


        #modified to repeat the mask across 3 channels for RGB images
        mask_weight = np.repeat(mask_weight[:,:, np.newaxis], 3, axis=2)
        #if mask_weight.shape[0] == 3:
        #    mask_weight = np.transpose(mask_weight, (1,2,0))
        
        #print("maskweight permuted ", str(mask_weight.shape))
        copyto = np.array((copyto * (1 - mask_weight)) + (copyfrom * mask_weight), dtype=np.float)

        copied_canvas += np.array(mask_weight*copyfrom)
        copied_canvas_to = np.zeros(np.shape(copyfrom))
        copied_canvas_to += np.array(mask_weight*copyto_ref)
        diff_copied = copied_canvas - copied_canvas_to
        
        fake_img_norm = normalize_image(copy.deepcopy(fake_img))
        fake_img_norm = torch.squeeze(image_to_tensor(fake_img_norm.astype(np.float32)))
        fake_img_norm_img = fake_img_norm.permute(2,0,1)
        #print(fake_img_norm.shape)
        #######figuring out noise without renorm
        #fakesy = fake_img_norm_img.cpu().detach().squeeze()
        #gradsfake = np.transpose(np.asarray(fakesy), (1,2,0))
        #im = Image.fromarray(gradsfake,'RGB')
        #im.save('fake_img_norenorm.png')
        #######figuring out noise with renorm
        
        #save_image(fake_img_norm_img, "fake_img_renorm.png")
        ###########This is fine
        #print(fake_img_norm_img.shape)
        #fake_img_norm = fake_img_norm_img.mean(0)
        #print(fake_img_norm.shape)
        out_fake =  F.softmax(net(fake_img_norm_img.unsqueeze(0)), dim=1)
        
        real_img_norm = normalize_image(copy.deepcopy(real_img))
        real_img_norm = torch.squeeze(image_to_tensor(real_img_norm.astype(np.float32)))
        real_img_norm_img = real_img_norm.permute(2,0,1)
        #real_img_norm = real_img_norm_img.mean(0)
        
        #print(real_img_norm.shape)
        out_real =  F.softmax(net(real_img_norm_img.unsqueeze(0)), dim=1)

        im_copied_norm = normalize_image(copy.deepcopy(copyto))
        im_copied_norm = torch.squeeze(image_to_tensor(im_copied_norm.astype(np.float32)))
        im_copied_norm_img = im_copied_norm.permute(2,0,1)
        #im_copied_norm = im_copied_norm_img.mean(0)
        
        #convert to grayscale
        #print(fake_img_norm.shape)
        out_copyto =  F.softmax(net(im_copied_norm_img.unsqueeze(0)), dim=1)        
        #print("real ", str(out_real))
        #print("fake", str(out_fake))
        #print("copy ", str(out_copyto))
        #print(fake_img_norm.shape)
        #print(attribution.shape)
        imgs = [attribution, real_img_norm, fake_img_norm, im_copied_norm, normalize_image(copied_canvas), 
                normalize_image(copied_canvas_to), normalize_image(diff_copied), mask_weight]

        imgs_all.append(imgs)

        mrf_score = out_copyto[0][real_class] - out_fake[0][real_class]     
        result_dict[thr] = [float(mrf_score.detach().cpu().numpy()), mask_size]

    return result_dict, img_names, imgs_all
