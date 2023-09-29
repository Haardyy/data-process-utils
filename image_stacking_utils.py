import numpy

from enum import Enum


class VerticalAlignment(Enum):
    LEFT = 'left'
    RIGHT = 'right'
    CENTER = 'center'


class HorizontalAlignment(Enum):
    TOP = 'top'
    BOTTOM = 'bottom'
    CENTER = 'center'


def combine_images_checkups(first_image: numpy.ndarray, second_image: numpy.ndarray):
    """Check up method for input parameters of functions: combine_images_horizontally and combine_images_vertically."""
    if first_image.dtype != second_image.dtype:
        raise TypeError("Different dtype between images. First_image",
                        first_image.dtype, "- second_image", second_image.dtype)

    if len(first_image.shape) <= 1 or len(first_image.shape) > 3:
        raise ValueError("Unsupported dimensionality for first image:",
                         str(len(first_image)) + ". Supported dimensions are 2D or 3D.")

    if len(second_image.shape) <= 1 or len(second_image.shape) > 3:
        raise ValueError("Unsupported dimensionality for second image:",
                         str(len(second_image)) + ". Supported dimensions are 2D or 3D.")

    if len(first_image.shape) != len(second_image.shape):
        raise ValueError("Images need to have same dimensionality. First_image:",
                         first_image.shape, "- second_image:", second_image.shape)

    if len(first_image.shape) == 3 and len(second_image.shape) == 3:
        if first_image.shape[2] != second_image.shape[2]:
            raise ValueError("Images need to have same number of channels.",
                             "Expected image shapes (Height, Width, Channels)")


def combine_images_horizontally(top_image: numpy.ndarray, bottom_image: numpy.ndarray):
    """Function will horizontally combine two numpy images into single one.

        Parameters
        ----------
        top_image : numpy.ndarray
            Image will be on the top side in final image. Expected image format (Height, Width) or
            (Height, Width, Channels).
        bottom_image : numpy.ndarray
            Image will be on the bottom side in final image. Expected image format (Height, Width) or
            (Height, Width, Channels).

        Returns
        -------
        combined_image : numpy.ndarray
            Image created from combination of images with shape(sum(height), max(width)) from both images.

        Raises
        ------
        TypeError
            If images have different dtype.
        ValueError
            If dimensions of images aren't 2D or 3D.
            If dimension between images are different.
            If images have different number of channels.
    """

    combine_images_checkups(first_image=top_image, second_image=bottom_image)

    top_height, top_width = top_image.shape[0], top_image.shape[1]
    bottom_height, bottom_width = bottom_image.shape[0], bottom_image.shape[1]
    dimensions = top_image.shape[2]

    if dimensions == 3:
        combined_image = numpy.zeros(shape=(top_height + bottom_height,
                                            max(top_width, bottom_width),
                                            top_image.shape[2]), dtype=top_image.dtype)

        combined_image[0:top_height, 0:top_width, :] = top_image
        combined_image[top_height:, 0:bottom_width, :] = bottom_image

    else:
        combined_image = numpy.zeros(shape=(top_height + bottom_height,
                                            max(top_width, bottom_width)), dtype=top_image.dtype)

        combined_image[0:top_height, 0:top_width] = top_image
        combined_image[top_height:, 0:bottom_width] = bottom_image

    return combined_image


def combine_images_vertically(left_image: numpy.ndarray, right_image: numpy.ndarray):
    """Function will vertically combine two numpy images into single one.

    Parameters
    ----------
    left_image : numpy.ndarray
        Image will be on the left side in final image. Expected image format (Height, Width) or
        (Height, Width, Channels).
    right_image : numpy.ndarray
        Image will be on the right side in final image. Expected image format (Height, Width) or
        (Height, Width, Channels).

    Returns
    -------
    combined_image : numpy.ndarray
        Image created from combination of images with shape(max(height), sum(width)) from both images.

    Raises
    ------
    TypeError
        If images have different dtype.
    ValueError
        If dimensions of images aren't 2D or 3D.
        If dimension between images are different.
        If images have different number of channels.
    """

    combine_images_checkups(first_image=left_image, second_image=right_image)

    left_height, left_width = left_image.shape[0], left_image.shape[1]
    right_height, right_width = right_image.shape[0], right_image.shape[1]
    dimensions = left_image.shape[2]

    if dimensions == 3:
        combined_image = numpy.zeros(shape=(max(left_height, right_height),
                                            left_width + right_width,
                                            left_image.shape[2]), dtype=left_image.dtype)

        combined_image[0:left_height, 0:left_width, :] = left_image
        combined_image[0:right_height, left_width:, :] = right_image

    else:
        combined_image = numpy.zeros(shape=(max(left_height, right_height),
                                            left_width + right_width), dtype=left_image.dtype)

        combined_image[0:left_height, 0:left_width] = left_image
        combined_image[0:right_height, left_width] = right_image

    return combined_image
