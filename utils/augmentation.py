import numpy as np
import skimage
from PIL import Image
from skimage.transform._warps_cy import _warp_fast
import config


def fast_warp(img, tf, output_shape, mode='constant', order=0):
    """
    This wrapper function is faster than skimage.transform.warp
    """
    m = tf.params
    t_img = np.zeros((img.shape[0],) + output_shape, img.dtype)
    for i in range(t_img.shape[0]):
        t_img[i] = _warp_fast(img[i], m, output_shape=output_shape,
                              mode=mode, order=order)
    return t_img


def load_augment(fname, w, h, aug_params=config.no_augmentation_params,
                 transform=None, sigma=0.0, color_vec=None):
    """Load augmented image with output shape (w, h).
    Default arguments return non augmented image of shape (w, h).
    To apply a fixed transform (color augmentation) specify transform
    (color_vec).
    To generate a random augmentation specify aug_params and sigma.
    """
    img = load_image(fname)
    if transform is None:
        img = perturb(img, augmentation_params=aug_params, target_shape=(w, h))
    else:
        img = perturb_fixed(img, tform_augment=transform, target_shape=(w, h))

    np.subtract(img, config.MEAN[:, np.newaxis, np.newaxis], out=img)
    np.divide(img, config.STD[:, np.newaxis, np.newaxis], out=img)
    img = augment_color(img, sigma=sigma, color_vec=color_vec)
    return img


def load_image(fname, w=config.raw_width, h=config.raw_height):
    if isinstance(fname, str):
        img = Image.open(fname)
        resized = img.resize([w, h])
        return np.array(resized, dtype=np.float32).transpose(2, 1, 0)
    else:
        return np.array([load_image(f) for f in fname])


def augment_color(img, sigma=0.1, color_vec=None):
    if color_vec is None:
        if not sigma > 0.0:
            color_vec = np.zeros(3, dtype=np.float32)
        else:
            color_vec = np.random.normal(0.0, sigma, 3)

    alpha = color_vec.astype(np.float32) * config.EV
    noise = np.dot(config.U, alpha.T)
    return img + noise[:, np.newaxis, np.newaxis]


def perturb(img, augmentation_params, target_shape, rng=np.random):
    # # DEBUG: draw a border to see where the image ends up
    # img[0, :] = 0.5
    # img[-1, :] = 0.5
    # img[:, 0] = 0.5
    # img[:, -1] = 0.5
    shape = img.shape[1:]
    tform_centering = build_centering_transform(shape, target_shape)
    tform_center, tform_uncenter = build_center_uncenter_transforms(shape)
    tform_augment = random_perturbation_transform(rng=rng, **augmentation_params)
    tform_augment = tform_uncenter + tform_augment + tform_center # shift to center, augment, shift back (for the rotation/shearing)
    return fast_warp(img, tform_centering + tform_augment,
                     output_shape=target_shape,
                     mode='constant')


# for transform augmentation
def perturb_fixed(img, tform_augment, target_shape=(50, 50)):
    shape = img.shape[1:]
    tform_centering = build_centering_transform(shape, target_shape)
    tform_center, tform_uncenter = build_center_uncenter_transforms(shape)
    tform_augment = tform_uncenter + tform_augment + tform_center # shift to center, augment, shift back (for the rotation/shearing)
    return fast_warp(img, tform_centering + tform_augment,
                     output_shape=target_shape, mode='constant')


def build_centering_transform(image_shape, target_shape):
    rows, cols = image_shape
    trows, tcols = target_shape
    shift_x = (cols - tcols) / 2.0
    shift_y = (rows - trows) / 2.0
    return skimage.transform.SimilarityTransform(translation=(shift_x, shift_y))


def build_center_uncenter_transforms(image_shape):
    """
    These are used to ensure that zooming and rotation happens around the center of the image.
    Use these transforms to center and uncenter the image around such a transform.
    """
    center_shift = np.array(
        [image_shape[1], image_shape[0]]) / 2.0 - 0.5  # need to swap rows and cols here apparently! confusing!
    tform_uncenter = skimage.transform.SimilarityTransform(translation=-center_shift)
    tform_center = skimage.transform.SimilarityTransform(translation=center_shift)
    return tform_center, tform_uncenter


def random_perturbation_transform(zoom_range, rotation_range, shear_range, translation_range, do_flip=True,
                                  allow_stretch=False, rng=np.random):
    shift_x = rng.uniform(*translation_range)
    shift_y = rng.uniform(*translation_range)
    translation = (shift_x, shift_y)

    rotation = rng.uniform(*rotation_range)
    shear = rng.uniform(*shear_range)

    if do_flip:
        flip = (rng.randint(2) > 0)  # flip half of the time
    else:
        flip = False

    # random zoom
    log_zoom_range = [np.log(z) for z in zoom_range]
    if isinstance(allow_stretch, float):
        log_stretch_range = [-np.log(allow_stretch), np.log(allow_stretch)]
        zoom = np.exp(rng.uniform(*log_zoom_range))
        stretch = np.exp(rng.uniform(*log_stretch_range))
        zoom_x = zoom * stretch
        zoom_y = zoom / stretch
    elif allow_stretch is True:  # avoid bugs, f.e. when it is an integer
        zoom_x = np.exp(rng.uniform(*log_zoom_range))
        zoom_y = np.exp(rng.uniform(*log_zoom_range))
    else:
        zoom_x = zoom_y = np.exp(rng.uniform(*log_zoom_range))
    # the range should be multiplicatively symmetric, so [1/1.1, 1.1] instead of [0.9, 1.1] makes more sense.

    return build_augmentation_transform((zoom_x, zoom_y), rotation, shear, translation, flip)


def build_augmentation_transform(zoom=(1.0, 1.0), rotation=0, shear=0, translation=(0, 0), flip=False):
    if flip:
        shear += 180
        rotation += 180
        # shear by 180 degrees is equivalent to rotation by 180 degrees + flip.
        # So after that we rotate it another 180 degrees to get just the flip.

    tform_augment = skimage.transform.AffineTransform(scale=(1 / zoom[0], 1 / zoom[1]), rotation=np.deg2rad(rotation),
                                                      shear=np.deg2rad(shear), translation=translation)
    return tform_augment
