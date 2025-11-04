# refactored module (auto-generated)


# ---- imports from original modules ----
from matplotlib import image

import torch



def combine_density_views(working_dir=None,n_times=None,n_views=None):
    """
    The purpose of this loop is to construct a structured tensor
    from a set of image files that are indexed by time step and view. At
    each time step, it loads a separate image for each view, extracts
    the red channel, and stacks these channel slices into a single tensor for
    that time step. Once all views for all time steps are processed, it combines
    the resulting time-step tensors into a final tensor. The output tensor thus has
    a shape [n_times, n_views, height, width], making it easy to access any view’s image
    data at any time step for further processing.
    """
    im_list=[]
    for tt in range(n_times):
        im_t_list=[]
        for ii_index in range(len(n_views)):
            ii = n_views[ii_index]
            #im = torch.tensor(image.imread("./traj_imgs/img_{}_{}.png".format(ii,tt)))[:,:,:3]
            im = torch.tensor(image.imread(working_dir+"/traj_imgs/img_{}_{}.png".format(ii,tt)))[:,:,:3]
            blur_im=im[:,:,0]
            im_t_list.append(blur_im.detach().clone()[None])
            im_list_tor=torch.vstack(im_t_list)[None]
        im_list.append(im_list_tor)
    im_list_tor_all=torch.vstack(im_list)
    return im_list_tor_all
