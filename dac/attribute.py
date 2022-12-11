from captum.attr import IntegratedGradients, Saliency, DeepLift,\
                        GuidedGradCam, InputXGradient,\
                        DeepLift, LayerGradCam, GuidedBackprop
import torch
import numpy as np
import os
import scipy
import scipy.ndimage
import sys

from dac.utils import save_image, normalize_image, image_to_tensor
from dac.activations import project_layer_activations_to_input_rescale
from dac.stereo_gc import get_sgc
from dac_networks import init_network

import pathlib
import platform

from fastai.learner import load_learner
from PIL import Image
###############################################################################
# Helper Functions
###############################################################################
def is_PDAC(x):
    if "ductal adenocarcinoma" in x:
        return True
    else:
        return False

plat = platform.system()
if plat == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

torch.manual_seed(123)
np.random.seed(123)

def get_attribution(real_img, 
                    fake_img, 
                    real_class, 
                    fake_class, 
                    net_module, 
                    checkpoint_path, 
                    input_shape, 
                    channels,
                    methods=["ig", "grads", "gc", "ggc", "dl", "ingrad", "random", "residual"],
                    output_classes=6,
                    downsample_factors=None,
                    bidirectional=False):

    '''Return (discriminative) attributions for an image pair.

    Args:

        real_img: (''array like'')
                
            Real image to run attribution on.


        fake_img: (''array like'')

            Counterfactual image typically created by a cycle GAN.

        real_class: (''int'')

            Class index of real image. Must correspond to networks output class.

        fake_class: (''int'')

            Class index of fake image. Must correspond to networks output class.

        net_module: (''str'')

            Name of network to use. Network is assumed to be specified at
            networks/{net_module}.py and have a matching class name.

        checkpoint_path: (''str'')

            Path to network checkpoint

        input_shape: (''tuple of int'')

            Spatial input image shape, must be 2D.

        channels: (''int'')

            Number of input channels

        methods: (''list of str'')

            List of attribution methods to run

        output_classes: (''int'')

            Number of network output classes

        downsample_factors: (''List of tuple of int'')

            Network argument specifying downsample factors

        bidirectional: (''int'')

            Return both attribution directions.
    '''

    imgs = [torch.transpose(torch.squeeze(image_to_tensor(normalize_image(real_img).astype(np.float32))), 2,0), 
                          torch.transpose(torch.squeeze(image_to_tensor(normalize_image(fake_img).astype(np.float32))), 2,0)]
    #print(imgs[0].shape)

    
    classes = [real_class, fake_class]
    #net = init_network(checkpoint_path, input_shape, net_module, channels, output_classes=output_classes,eval_net=True, require_grad=False,
    #                   downsample_factors=downsample_factors)
    device = torch.device('cuda')
    learner = load_learner(checkpoint_path)
    net= learner.model
    net.to(device)
    attrs = []
    attrs_names = []

    if "residual" in methods:
        #print("residual called ")
        #print(np.max(real_img), np.min(real_img), np.max(fake_img), np.min(fake_img))
        res = real_img - fake_img
        #print(np.max(res), np.min(res))
        res = np.abs(res)
        #save_image(res, "ressubtracted.png", renorm=False)
        #print(np.max(res), np.min(res))
        #print(res.dtype)
        #print(np.min(res)) these are 255 and 0, which is wrong
        #print(np.max(res))
        res = res - np.min(res)
        res = torch.tensor(res/np.max(res))
        #print(res)
        #print('res', str(res.shape)) # [256,256,3]
        #res = res.permute(2,0,1)
        #save_image(real_img, 'real.png', renorm=False)
        #save_image(fake_img, 'fake.png', renorm=False)
        #resgray = np.repeat(np.mean(np.asarray(res), axis=2)[:,:,np.newaxis], 3, axis=2)
        #print(resgray.shape)
        #print(resgray[0:10, 0:10, 0:3])
        #resgray = (resgray*255).astype(np.uint8)
        #print("maxresgray", str(np.max(resgray)))
        #print("minresgray", str(np.min(resgray)))
        #im = Image.fromarray(resgray,'RGB')
        #im.save('resgraydirectsave.png')
        #save_image(resgray, 'resgray.png', renorm=False)
        #save_image((res*255), "res.png", renorm=False)
        attrs.append(res)
        attrs_names.append("residual")

    if "random" in methods:
        rand = np.abs(np.random.randn(*np.shape(real_img)))
        rand = np.abs(scipy.ndimage.filters.gaussian_filter(rand, 4))
        rand = rand - np.min(rand)
        rand = rand/np.max(np.abs(rand))
        attrs.append(torch.tensor(rand).permute(2,0,1))
        attrs_names.append("random")

    if "gc" in methods:
        net.zero_grad()
        last_conv_layer = [(name,module) for name, module in net.named_modules() if type(module) == torch.nn.Conv2d][-1]
        layer_name = last_conv_layer[0]
        layer = last_conv_layer[1]
        layer_gc = LayerGradCam(net, layer)

        gc_real = layer_gc.attribute(torch.unsqueeze(imgs[0],0), target=classes[0])

        gc_real = project_layer_activations_to_input_rescale(gc_real.cpu().detach().numpy(), (input_shape[0], input_shape[1]))
        #print("gc_real")
        #print(gc_real)
        attrs.append(torch.tensor(gc_real[0,0,:,:]))
        attrs_names.append("gc")

        gc_diff_0, gc_diff_1 = get_sgc(real_img, fake_img, real_class, 
                                     fake_class, net_module, checkpoint_path, 
                                     input_shape, channels, None, output_classes=output_classes,
                                     downsample_factors=downsample_factors)
        attrs.append(gc_diff_0)
        attrs_names.append("d_gc")

        if bidirectional:
            gc_fake = layer_gc.attribute(imgs[1], target=classes[1])
            gc_fake = project_layer_activations_to_input_rescale(gc_fake.cpu().detach().numpy(), (input_shape[0], input_shape[1]))
            attrs.append(torch.tensor(gc_fake[0,0,:,:]))
            attrs_names.append("gc_fake")

            attrs.append(gc_diff_1)
            attrs_names.append("d_gc_inv")

    if "ggc" in methods:
        net.zero_grad()
        last_conv = [module for module in net.modules() if type(module) == torch.nn.Conv2d][-1]

        # Real
        guided_gc = GuidedGradCam(net, last_conv)
        ggc_real = guided_gc.attribute(imgs[0], target=classes[0])
        attrs.append(ggc_real[0,0,:,:])
        attrs_names.append("ggc")

        gc_diff_0, gc_diff_1 = get_sgc(real_img, fake_img, real_class, 
                                     fake_class, net_module, checkpoint_path, 
                                     input_shape, channels, None, output_classes=output_classes,
                                     downsample_factors=downsample_factors)

        # D-gc
        net.zero_grad()
        gbp = GuidedBackprop(net)
        gbp_real = gbp.attribute(imgs[0], target=classes[0])
        ggc_diff_0 = gbp_real[0,0,:,:] * gc_diff_0
        attrs.append(ggc_diff_0)
        attrs_names.append("d_ggc")

        if bidirectional:
            ggc_fake = guided_gc.attribute(imgs[1], target=classes[1])
            attrs.append(ggc_fake[0,0,:,:])
            attrs_names.append("ggc_fake")

            gbp_fake = gbp.attribute(imgs[1], target=classes[1])
            ggc_diff_1 = gbp_fake[0,0,:,:] * gc_diff_1
            attrs.append(ggc_diff_1)
            attrs_names.append("d_ggc_inv")

    # IG
    if "ig" in methods:
        baseline = image_to_tensor(np.zeros(input_shape, dtype=np.float32))
        net.zero_grad()
        ig = IntegratedGradients(net)
        ig_real, delta_real = ig.attribute(imgs[0].unsqueeze(0), baseline, target=classes[0], return_convergence_delta=True)
        ig_diff_1, delta_diff = ig.attribute(imgs[1], imgs[0].unsqueeze(0), target=classes[1], return_convergence_delta=True)

        attrs.append(ig_real[0,:,:,:])
        attrs_names.append("ig")

        attrs.append(ig_diff_1[0,:,:,:])
        attrs_names.append("d_ig")

        if bidirectional:
            ig_fake, delta_fake = ig.attribute(imgs[1], baseline, target=classes[1], return_convergence_delta=True)
            attrs.append(ig_fake[0,0,:,:])
            attrs_names.append("ig_fake")

            ig_diff_0, delta_diff = ig.attribute(imgs[0], imgs[1], target=classes[0], return_convergence_delta=True)
            attrs.append(ig_diff_0[0,0,:,:])
            attrs_names.append("d_ig_inv")

        
    # DL
    if "dl" in methods:
        net.zero_grad()
        dl = DeepLift(net)
        print('imgs[0]', str(imgs[0].shape))
        dl_real = dl.attribute(imgs[0].unsqueeze(0), target=classes[0])
        dl_diff_1 = dl.attribute(imgs[1].unsqueeze(0), baselines=imgs[0].unsqueeze(0), target=classes[1])

        attrs.append(dl_real[:,:,:])
        attrs_names.append("dl")

        attrs.append(dl_diff_1[:,:,:])
        attrs_names.append("d_dl")

        if bidirectional:
            dl_fake = dl.attribute(imgs[1], target=classes[1])
            attrs.append(dl_fake[:,:,:])
            attrs_names.append("dl_fake")

            dl_diff_0 = dl.attribute(imgs[0], baselines=imgs[1], target=classes[0])
            attrs.append(dl_diff_0[:,:,:])
            attrs_names.append("d_dl_inv")

    # INGRAD
    if "ingrad" in methods:
        #print("ingrad called:")
        net.zero_grad()
        saliency = Saliency(net)
        grads_real = saliency.attribute(torch.unsqueeze(imgs[0],0), 
                                        target=classes[0]) 
        grads_fake = saliency.attribute(torch.unsqueeze(imgs[1],0), 
                                        target=classes[1])
        ##### Noise)
        #gradsfake = grads_fake.cpu().detach().squeeze()
        #gradsfake = np.transpose(np.asarray(gradsfake), (1,2,0))
        #im = Image.fromarray(gradsfake,'RGB')
        #im.save('gradsfake.png')
        ###########This is noise
        net.zero_grad()
        input_x_gradient = InputXGradient(net)
        ingrad_real = (input_x_gradient.attribute(torch.unsqueeze(imgs[0],0), target=classes[0])).squeeze()
        #print("ingrad_real "+str(ingrad_real.shape))
        #print("grads_fake "+str(grads_fake.shape))
        #print("imgs[0] "+str(imgs[0].shape))
        ingrad_diff_0 = grads_fake.squeeze() * (imgs[0] - imgs[1])
        #print("ingrad_diff_0 "+ str(ingrad_diff_0.shape))
        # [3,256,256]
        #######figuring out noise
        #export = torch.abs(ingrad_real[:,:,:]).cpu().detach()
        #export = np.asarray(export)
        #export = np.transpose(export, (1,2,0))
        #im = Image.fromarray(export,'RGB')
        #im.save('ingrad_real.png')
        ###########This is noise
        
        attrs.append(torch.abs(ingrad_real[:,:,:]))
        attrs_names.append("ingrad")
        #print("attrs shape", str(attrs[-1].shape))
        attrs.append(torch.abs(ingrad_diff_0[:,:,:]))
        attrs_names.append("d_ingrad")

        if bidirectional:
            print("bidirectionalcheck")
            ingrad_fake = input_x_gradient.attribute(torch.unsqueeze(imgs[1],0), target=classes[1])
            attrs.append(torch.abs(ingrad_fake[0,0,:,:]))
            attrs_names.append("ingrad_fake")

            ingrad_diff_1 = grads_real * (imgs[1] - imgs[0])
            attrs.append(torch.abs(ingrad_diff_1[0,0,:,:]))
            attrs_names.append("d_ingrad_inv")

    attrs = [a.detach().cpu().numpy() for a in attrs]
    #check attrs shape
    #print("attrs[0] ", str(attrs[0].shape))
    attrs_norm = [a/np.max(np.abs(a)) for a in attrs]

    return attrs_norm, attrs_names
