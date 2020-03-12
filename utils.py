import numpy as np
import cv2
import random
from PIL import Image


class Image(object):
    def __init__(self, path):
        self.path = path
        self.bgr_img = None
        self.gray_img = None
        self.rgb_img = None

    def read_image(self, return_img=False):
        self.bgr_img = cv2.imread(self.path)
        if return_img:
            return self.bgr_img

    def rgb(self, return_img=False):
        self.rgb_img = cv2.cvtColor(self.bgr_img, cv2.COLOR_BGR2RGB)
        if return_img:
            return self.rgb_img

    def gray(self, return_img=False):
        self.gray_img = cv2.cvtColor(self.bgr_img, cv2.COLOR_BGR2GRAY)
        if return_img:
            return self.gray_img

    def show(self, img, title='image'):
        if len(img.shape) != 3:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)
        plt.title(title)

    def show_all(self, image_list, title_list):
        plt.figure(figsize=[20, 10])
        assert len(image_list) == len(title_list), "Houston we have a problem"
        N = len(image_list)
        for index, (img, title) in enumerate(zip(image_list, title_list)):
            plt.subplot(1, N, index + 1)
            if len(img.shape) != 3:
                plt.imshow(img, cmap='gray')
            else:
                plt.imshow(img)
            plt.title(title)
        plt.show()


class Aug:

    @staticmethod
    def _normalize(image):
        """
        Normalizes the image into 0-255 unsigned 8 bit integer
        """
        return cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    @staticmethod
    def _convolve2d(image, kernel):
        """
        Convolves im with kernel, over all three colour channels
        """
        return cv2.filter2D(image, -1, kernel)

    @staticmethod
    def flip(image):
        """
        flips an image vertically
        """
        new_image = np.flipud(image)
        return new_image.astype('uint8')

    @staticmethod
    def mirror(image):
        """
        mirrors an image horizontally
        """
        new_image = np.fliplr(image)
        return new_image.astype('uint8')

    @staticmethod
    def _r_c(x, y, theta):
        """
        Returns a tuple of new matrix containing rotated coords

        Parameters:
            x (np.array): Coords matrix
            y (np.array): Coords matrix
            theta (float): Amount of rotation in radians

        Returns:
            The augmented image
        """
        new_x = x * np.cos(theta) - y * np.sin(theta)
        new_y = x * np.sin(theta) + y * np.cos(theta)
        return new_x, new_y

    @staticmethod
    def rotate_bound(image, angle=None):
        """
        Rotates the image by angle and keeps then original image shape

        Parameters:
            image (2D array): Input image
            angle (float): Angle of rotation in radians

        Returns:
            The augmented image
        """
        if angle is None:
            angle = (np.random.random() / 2 - 0.25)
        angle = np.degrees(angle)
        rows, cols = image.shape[0:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        image_rotate = cv2.warpAffine(image, M, (cols, rows))

        return image_rotate.astype('uint8')

    @staticmethod
    def rotate(image, angle=None):
        """
        Rotates the image by angle and updates the image shape to fit the new rotated image

        Parameters:
            image (2D array): Input image
            angle (float): Angle of rotation in radians

        Returns:
            The augmented image
        """
        if angle is None:
            angle = (np.random.rand(1) - 0.5)
        if len(image.shape) == 3:
            rows, cols, ch = image.shape
        elif len(image.shape) == 2:
            rows, cols = image.shape
        x_coords, y_coords = Aug._r_c(np.array([0, cols, cols, 0]), np.array([0, 0, rows, rows]), angle)

        new_max_x = int(np.ceil(x_coords.max() - x_coords.min()))
        new_max_y = int(np.ceil(y_coords.max() - y_coords.min()))

        x, y = np.meshgrid(np.arange(new_max_x), np.arange(new_max_y))
        xn, yn = Aug._r_c(x + x_coords.min(), y + y_coords.min(), -angle)

        xn = xn.astype(int)
        yn = yn.astype(int)

        mask = (xn >= 0) & (xn < cols) & (yn >= 0) & (yn < rows)

        if len(image.shape) == 3:
            image_rotate = np.zeros([new_max_y, new_max_x, 3], dtype=image.dtype)
        elif len(image.shape) == 2:
            image_rotate = np.zeros([new_max_y, new_max_x], dtype=image.dtype)

        image_rotate[y[mask], x[mask]] = image[yn[mask], xn[mask]]
        image_rotate[y[~mask], x[~mask]] = 0

        return image_rotate.astype('uint8')

    @staticmethod
    def gaussian_blur(image, amount=None):
        """
        Applies a gaussian blur filter on the image

        Parameters:
            image (2D array): Input image
            amount (int): Controls how strong the effect is

        Returns:
            The augmented image
        """
        if amount is None:
            amount = np.random.randint(3, 11) * 2 + 1
        return cv2.GaussianBlur(image, (amount, amount), 0)

    @staticmethod
    def box_blur(image, kernel_size=7):
        """
        Applies a box blur filter on the image

        Parameters:
            image (2D array): Input image
            kernel_size (int): Determines the size of the kernel (default 7x7)

        Returns:
            The augmented image
        """
        kernel = np.ones((kernel_size, kernel_size))
        kernel /= np.sum(kernel)
        output = Aug._convolve2d(image, kernel)
        return output

    @staticmethod
    def median_blur(image, filtersize=3):
        """
        Applies a median blur on the image

        Parameters:
            image (2D array): Input image
            filtersize (int): The size of the passing filter on the image (default 3x3)

        Returns:
            The augmented image
        """
        output = np.zeros_like(image, dtype=image.dtype)
        padding = filtersize // 2
        index = (filtersize ** 2) // 2
        if len(image.shape) == 3:
            image_padded = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), 'reflect')
            for channel in range(3):
                for x in range(image.shape[1]):
                    for y in range(image.shape[0]):
                        output[y, x, channel] = \
                            np.sort(image_padded[y:y + filtersize, x:x + filtersize, channel], axis=None)[index]
        elif len(image.shape) == 2:
            image_padded = np.pad(image, padding, 'reflect')
            for x in range(image.shape[1]):
                for y in range(image.shape[0]):
                    output[y, x] = np.sort(image_padded[y:y + filtersize, x:x + filtersize], axis=None)[index]
        return (np.clip(output, 0, 255)).astype('uint8')

    @staticmethod
    def zoom(image, amount=None):
        """
        Zooms the image by a given amount
        """
        if amount is None:
            amount = np.random.rand(1) + 1
        mid_x = image.shape[1] // 2
        mid_y = image.shape[0] // 2
        xz = int(image.shape[1] / (amount * 2))
        yz = int(image.shape[0] / (amount * 2))
        zoomed_img = image[mid_y - yz:mid_y + yz, mid_x - xz:mid_x + xz]
        output = cv2.resize(zoomed_img, (image.shape[1], image.shape[0]), cv2.INTER_LINEAR)
        return output.astype('uint8')

    @staticmethod
    def crop(image, amount=None):
        """
        Crops the image by a given amount
        """
        if amount is None:
            amount = np.random.rand(1) + 1
        mid_x = image.shape[1] // 2
        mid_y = image.shape[0] // 2
        xz = int(image.shape[1] / (amount * 2))
        yz = int(image.shape[0] / (amount * 2))
        cropped_img = image[mid_y - yz:mid_y + yz, mid_x - xz:mid_x + xz]
        return cropped_img
    
    @staticmethod
    def shear_image(image, shear = 0.2):
        from skimage import transform 
        afine = transform.AffineTransform(shear = shear)
        modified = transform.warp(image,afine)
        return modified

    @staticmethod
    def invert(image):
        """
        Inverts the image
        """
        image = 255 - image
        return image

    @staticmethod
    def gauss_noise(image, sigma=None, mean=0):
        """
        Add gaussian noise to the image with the specified Sigma and Mean
        """
        if sigma is None:
            sigma = np.random.randint(5, 20)
        if len(image.shape) == 3:
            rows, cols, ch = image.shape
        elif len(image.shape) == 2:
            rows, cols = image.shape
        var = sigma ** 2
        gauss = np.random.normal(mean, sigma, (rows, cols, ch))
        gauss = gauss.reshape(rows, cols, ch)
        noisy = image + gauss
        return (np.clip(noisy, 0, 255)).astype('uint8')

    @staticmethod
    def snp_noise(image, s_vs_p=0.5, amount=0.004):
        """
        Add salt and pepper noise to the image

        Parameters:
            image (2D array): Input image
            s_vs_p (float): determines the ration between salt and pepper
            amount (float): controls how much to apply

        Returns:
            The augmented image
        """
        rows, cols = image.shape[0:2]
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coordsy, coordsx = np.random.randint(0, rows, int(num_salt)), np.random.randint(0, cols, int(num_salt))
        out[coordsy, coordsx] = 255

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coordsy, coordsx = np.random.randint(0, rows, int(num_pepper)), np.random.randint(0, cols, int(num_pepper))
        out[coordsy, coordsx] = 0
        return (np.clip(out, 0, 255)).astype('uint8')

    @staticmethod
    def translation(image, dx=10, dy=10):
        """
        Moves the image by a given amount

        Parameters:
            image (2D array): Input image
            dx (int): Range in pixels to move in x axis
            dy (int): Range in pixels to move in y axis

        Returns:
            The augmented image
        """
        
        dx = np.random.randint(dx)
        positive = random.choice([-1, 1])
        dx *= positive
        
        dy = np.random.randint(dy)
        positive = random.choice([-1, 1])
        dy *= positive
        
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        out = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        return out.astype('uint8')

    @staticmethod
    def sharpen(image):
        """
        Sharpens the image
        """
        kernel_sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        output = Aug._convolve2d(image, kernel_sharpen)
        return output

    @staticmethod
    def emboss(image):
        """
        Applies emboss on the image
        """
        kernel_emboss = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        output = Aug._convolve2d(image, kernel_emboss)
        return output

    @staticmethod
    def outline(image):
        """
        Applies an outline filter on the image
        """
        kernel_outline = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        output = Aug._convolve2d(image, kernel_outline)
        return output

    @staticmethod
    def bottom_sobel(image):
        """
        Applies bottom sobel filter on the image
        """
        kernel_bottom_sobel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        output = Aug._convolve2d(image, kernel_bottom_sobel)
        return output

    @staticmethod
    def posterize(image, bits=4):
        """
        Lowers the image bit depth to a given number

        Parameters:
            image (2D array): Input image
            bits (int): The depth in bits of the output image (1-8)

        Returns:
            The augmented image
        """
        mask = ~(2 ** (8 - bits) - 1)
        return (image & mask).astype('uint8')

    @staticmethod
    def brightness_contrast(image, brightness=None, contrast=None):
        """
        Applies both brightness and contrast filter on the image.

        Parameters:
            image (2D array): Input image
            brightness (int): The brightness value to be added (-127,127).
            contrast (int): The contrast value to multiply the image by (-127,127).

        Returns:
            buf: The augmented image.
        """
        if brightness is None:
            brightness = np.random.randint(-50, 50)
        if contrast is None:
            contrast = np.random.randint(-50, 50)

        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow) / 255
            gamma_b = shadow

            buf = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)
        else:
            buf = image.copy()

        if contrast != 0:
            f = 131 * (contrast + 127) / (127 * (131 - contrast))
            alpha_c = f
            gamma_c = 127 * (1 - f)

            buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

        return buf

    @staticmethod
    def shift_color(image, amount=None):
        """
        Shifts the color of the image by a random/given amount.

        Parameters:
            image (2D array): Input image
            amount (int): The amount from which to shift the colors by random(-amount,amount).

        Returns:
            out: The augmented image.
        """
        if amount is None:
            amount = np.random.randint(5, 50)
        shift_by = np.random.randint(-amount, amount, 3, dtype='int16')
        out = image + shift_by
        return (np.clip(out, 0, 255)).astype('uint8')

    @staticmethod
    def apply_funcs(num_of_funcs, image, *args):
        """
        Receive number of functions to randomly apply and functions with probability
        args are tuples of (function, probability)

        Parameters:
            num_of_funcs (int): How many of given functions to apply on selected image
            image (2D array): Input image
            *args (tuple): Functions to randomly apply with probability (function, probability).
        """
        func_arr = []
        output_func_arr = []
        for func, probability in args:
            func_arr.append((func, probability))
        for i in range(num_of_funcs):
            (func, probability) = random.choice(func_arr)
            func_arr.remove((func, probability))
            output_func_arr.append((func, probability))
        for func, probability in output_func_arr:
            if np.random.rand(1) <= probability:
                image = func(image)
        return image
