import math
import numpy

from enum import Enum
from typing import Union, Tuple, List


class HorizontalAlignment(Enum):
    LEFT = 'left'
    RIGHT = 'right'
    CENTER = 'center'


class VerticalAlignment(Enum):
    TOP = 'top'
    BOTTOM = 'bottom'
    CENTER = 'center'


def stack_images_checkups(first_image: numpy.ndarray, second_image: numpy.ndarray):
    """Check up method for input parameters of functions: stack_images_horizontally and stack_images_vertically."""
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


def stack_multiple_images_horizontally(images: List[numpy.ndarray],
                                       background_color: Union[int, Tuple[int, int, int], Tuple[int, int, int, int]]
                                       = 0,
                                       horizontal_alignment: Union[str, HorizontalAlignment]
                                       = HorizontalAlignment.CENTER):
    """Simple cyclic function based on stack_images_horizontally."""

    if len(images) < 2:
        raise ValueError("List is empty or with single image.")

    stacked_image = images[0]
    for second_image in images[1:]:
        stacked_image = stack_images_horizontally(top_image=stacked_image, bottom_image=second_image,
                                                  background_color=background_color,
                                                  horizontal_alignment=horizontal_alignment)

    return stacked_image


def stack_images_horizontally(top_image: numpy.ndarray, bottom_image: numpy.ndarray,
                              background_color: Union[int, Tuple[int, int, int], Tuple[int, int, int, int]] = 0,
                              horizontal_alignment: Union[str, HorizontalAlignment] = HorizontalAlignment.CENTER):
    """Function will horizontally stack two numpy images into single one.

        Parameters
        ----------
        top_image : numpy.ndarray
            Image will be on the top side in final image. Expected image format (Height, Width) or
            (Height, Width, Channels).
        bottom_image : numpy.ndarray
            Image will be on the bottom side in final image. Expected image format (Height, Width) or
            (Height, Width, Channels).
        background_color: Union[int, (int, int, int), (int, int, int, int)]
            If images have different widths then rest of image will be filled with provided color (default is 0 - black)
        horizontal_alignment: Union[str, HorizontalAlignment]
            If images have different widths then smaller image will be horizontally aligned (default is center)
            Possible string values (Case-Insensitive): right, left, center

        Returns
        -------
        stacked_image : numpy.ndarray
            Image created from combination of images with shape(sum(height), max(width)) from both images.

        Raises
        ------
        TypeError
            If images have different dtype.
        ValueError
            If dimensions of images aren't 2D or 3D.
            If dimension between images are different.
            If images have different number of channels.
            If background color isn't int and have different number of channels than images
    """

    stack_images_checkups(first_image=top_image, second_image=bottom_image)

    if type(background_color) != int:
        if len(top_image.shape) == 2:
            raise ValueError("Grayscale image can not accept multi channel color. Background color needs to be int.")
        elif len(background_color) != top_image.shape[3]:
            raise ValueError("Background color needs to have same number of channels or to be int.")

    if type(horizontal_alignment) == str:
        horizontal_alignment = HorizontalAlignment(horizontal_alignment.lower())

    top_height, top_width = top_image.shape[0], top_image.shape[1]
    bottom_height, bottom_width = bottom_image.shape[0], bottom_image.shape[1]
    dimensions = len(top_image.shape)
    stacked_image_width = max(top_width, bottom_width)

    if dimensions == 3:
        stacked_image = numpy.zeros(shape=(top_height + bottom_height,
                                           stacked_image_width,
                                           top_image.shape[2]), dtype=top_image.dtype)

        if type(background_color) != int or background_color != 0:
            stacked_image[:, :] = background_color

        if horizontal_alignment == HorizontalAlignment.LEFT:
            stacked_image[0:top_height, 0:top_width, :] = top_image
            stacked_image[top_height:, 0:bottom_width, :] = bottom_image

        elif horizontal_alignment == HorizontalAlignment.CENTER:
            top_center_start = math.ceil((stacked_image_width / 2) - (top_width / 2))
            bottom_center_start = math.ceil((stacked_image_width / 2) - (bottom_width / 2))

            stacked_image[0:top_height, top_center_start:top_center_start + top_width, :] = top_image
            stacked_image[top_height:, bottom_center_start:bottom_center_start + bottom_width, :] = bottom_image

        elif horizontal_alignment == HorizontalAlignment.RIGHT:
            top_right_start = stacked_image_width - top_width
            bottom_right_start = stacked_image_width - bottom_width

            stacked_image[0:top_height, top_right_start:, :] = top_image
            stacked_image[top_height:, bottom_right_start:, :] = bottom_image

    else:
        stacked_image = numpy.zeros(shape=(top_height + bottom_height, stacked_image_width), dtype=top_image.dtype)

        if type(background_color) != int or background_color != 0:
            stacked_image[:, :] = background_color

        if horizontal_alignment == HorizontalAlignment.LEFT:
            stacked_image[0:top_height, 0:top_width] = top_image
            stacked_image[top_height:, 0:bottom_width] = bottom_image

        elif horizontal_alignment == HorizontalAlignment.CENTER:
            top_center_start = math.ceil((stacked_image_width / 2) - (top_width / 2))
            bottom_center_start = math.ceil((stacked_image_width / 2) - (bottom_width / 2))

            stacked_image[0:top_height, top_center_start:top_center_start + top_width] = top_image
            stacked_image[top_height:, bottom_center_start:bottom_center_start + bottom_width] = bottom_image

        elif horizontal_alignment == HorizontalAlignment.RIGHT:
            top_right_start = stacked_image_width - top_width
            bottom_right_start = stacked_image_width - bottom_width

            stacked_image[0:top_height, top_right_start:] = top_image
            stacked_image[top_height:, bottom_right_start:] = bottom_image

    return stacked_image


def stack_multiple_images_vertically(images: List[numpy.ndarray],
                                     background_color: Union[int, Tuple[int, int, int], Tuple[int, int, int, int]] = 0,
                                     vertical_alignment: Union[str, VerticalAlignment] = VerticalAlignment.CENTER):
    """Simple cyclic function based on stack_images_horizontally."""

    if len(images) < 2:
        raise ValueError("List is empty or with single image.")

    stacked_image = images[0]
    for second_image in images[1:]:
        stacked_image = stack_images_vertically(left_image=stacked_image, right_image=second_image,
                                                background_color=background_color,
                                                vertical_alignment=vertical_alignment)

    return stacked_image


def stack_images_vertically(left_image: numpy.ndarray, right_image: numpy.ndarray,
                            background_color: Union[int, Tuple[int, int, int], Tuple[int, int, int, int]] = 0,
                            vertical_alignment: Union[str, VerticalAlignment] = VerticalAlignment.CENTER):
    """Function will vertically stack two numpy images into single one.

    Parameters
    ----------
    left_image : numpy.ndarray
        Image will be on the left side in final image. Expected image format (Height, Width) or
        (Height, Width, Channels).
    right_image : numpy.ndarray
        Image will be on the right side in final image. Expected image format (Height, Width) or
        (Height, Width, Channels).
    background_color: Union[int, (int, int, int), (int, int, int, int)]
            If images have different heights then rest of image will be filled with provided
            color (default is 0 - black)
    vertical_alignment: Union[str, VerticalAlignment]
        If images have different heights then smaller image will be vertically aligned (default is center)
        Possible string values (Case-Insensitive): top, bottom, center

    Returns
    -------
    stacked_image : numpy.ndarray
        Image created from combination of images with shape(max(height), sum(width)) from both images.

    Raises
    ------
    TypeError
        If images have different dtype.
    ValueError
        If dimensions of images aren't 2D or 3D.
        If dimension between images are different.
        If images have different number of channels.
        If background color isn't int and have different number of channels than images
    """

    stack_images_checkups(first_image=left_image, second_image=right_image)

    if type(background_color) != int:
        if len(left_image.shape) == 2:
            raise ValueError("Grayscale image can not accept multi channel color. Background color needs to be int.")
        elif len(background_color) != left_image.shape[3]:
            raise ValueError("Background color needs to have same number of channels or to be int.")

    if type(vertical_alignment) == str:
        vertical_alignment = VerticalAlignment(vertical_alignment.lower())

    left_height, left_width = left_image.shape[0], left_image.shape[1]
    right_height, right_width = right_image.shape[0], right_image.shape[1]
    dimensions = len(left_image.shape)
    stacked_image_height = max(left_height, right_height)

    if dimensions == 3:
        stacked_image = numpy.zeros(shape=(stacked_image_height,
                                           left_width + right_width,
                                           left_image.shape[2]), dtype=left_image.dtype)

        if type(background_color) != int or background_color != 0:
            stacked_image[:, :] = background_color

        if vertical_alignment == vertical_alignment.TOP:
            stacked_image[0:left_height, 0:left_width, :] = left_image
            stacked_image[0:right_height, left_width:, :] = right_image

        elif vertical_alignment == vertical_alignment.CENTER:
            left_center_start = math.ceil((stacked_image_height / 2) - (left_height / 2))
            right_center_start = math.ceil((stacked_image_height / 2) - (right_height / 2))

            stacked_image[left_center_start:left_center_start + left_height, 0:left_width, :] = left_image
            stacked_image[right_center_start:right_center_start + right_height, left_width:, :] = right_image

        elif vertical_alignment == vertical_alignment.BOTTOM:
            left_bottom_start = stacked_image_height - left_height
            right_bottom_start = stacked_image_height - right_height

            stacked_image[left_bottom_start:, 0:left_width, :] = left_image
            stacked_image[right_bottom_start:, left_width:, :] = right_image

    else:
        stacked_image = numpy.zeros(shape=(stacked_image_height,
                                           left_width + right_width), dtype=left_image.dtype)

        if type(background_color) != int or background_color != 0:
            stacked_image[:, :] = background_color

        if vertical_alignment == vertical_alignment.TOP:
            stacked_image[0:left_height, 0:left_width] = left_image
            stacked_image[0:right_height, left_width:] = right_image

        elif vertical_alignment == vertical_alignment.CENTER:
            left_center_start = math.ceil((stacked_image_height / 2) - (left_height / 2))
            right_center_start = math.ceil((stacked_image_height / 2) - (right_height / 2))

            stacked_image[left_center_start:left_center_start + left_height, 0:left_width] = left_image
            stacked_image[right_center_start:right_center_start + right_height, left_width:] = right_image

        elif vertical_alignment == vertical_alignment.BOTTOM:
            left_bottom_start = stacked_image_height - left_height
            right_bottom_start = stacked_image_height - right_height

            stacked_image[left_bottom_start:, 0:left_width] = left_image
            stacked_image[right_bottom_start:, left_width:] = right_image

    return stacked_image
