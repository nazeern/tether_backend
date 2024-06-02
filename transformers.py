import numpy as np
from dataclasses import dataclass
from typing import ClassVar, Literal
from scipy.ndimage import shift
import cv2
from sklearn import decomposition
import multiprocess as mp
from typing import Any
from scipy.signal import convolve2d, medfilt2d


def join_sublists(
    sublists: list[Any]
):
    if isinstance(sublists[0], list):
        return [
            val
            for sublist in sublists
            for val in sublist
        ]
    else:
        return sublists


class Transformer:
    """
    A base class to abstract operations on a list of numpy arrays.
    Subclasses should implement the apply function that accepts a list of ndarrays.
    """

    optype = "transformation"

    def transform(
        self,
        arr: np.ndarray,
    ):
        raise NotImplementedError("Must implement transformation function to apply to an ndarray")
    

    def apply(
        self,
        arr_list: list[np.ndarray | str]
    ):
        num_arrs = len(arr_list)
        try:
            cpus = mp.cpu_count()
        except NotImplementedError:
            cpus = 2   # arbitrary default
        print(f"Performing {self.optype} of {num_arrs} arrays with {cpus} cpus...")
        
        with mp.Pool(processes=cpus) as pool:
            image_sublists = list(pool.imap(self.transform, arr_list))
        # image_sublists = map(self.transform, arr_list)
        
        return join_sublists(image_sublists)


@dataclass
class Loader(Transformer):
    
    optype: ClassVar[str] = "loading"
    
    def transform(
        self,
        image_path: str
    ):
        return cv2.imread(image_path)[...,::-1]


@dataclass
class Resizer(Transformer):
    """
    A class to handle image resizing.
    """
    optype: ClassVar[str] = "resizing"
    width: int
    height: int

    def transform(
        self,
        arr: np.ndarray
    ):
        return cv2.resize(arr, (self.width, self.height), cv2.INTER_NEAREST)

@dataclass
class Grayscaler(Transformer):
    """
    A class to handle image grayscaling.
    """
    optype: ClassVar[str] = "grayscaling"
    
    def transform(
        self,
        arr: np.ndarray
    ):
        return np.dot(arr, [0.2989, 0.5870, 0.1140])
        

@dataclass
class Translator(Transformer):
    """
    A class to handle image translation.
    """
    
    optype: ClassVar[str] = "translation"
    xmax: int = 0
    xnum: int = 1
    ymax: int = 0
    ynum: int = 1
    mode: Literal["crop", "replicate", "zeros", "roll"] = "replicate"

    def xvals(self):
        return [
            round(self.xmax * (x / self.xnum)) 
            for x 
            in range(-self.xnum, self.xnum + 1) 
            if x != 0
        ]

    
    def yvals(self):
        return [
            round(self.ymax * (y / self.ynum)) 
            for y 
            in range(-self.ynum, self.ynum + 1) 
            if y != 0
        ]


    def set_borders(
        arr: np.ndarray,
        shift: tuple[int, int] = (0, 0)
    ):
        xshift, yshift = shift
        if yshift > 0:
            arr[:yshift] = 0
        elif yshift < 0:
            arr[yshift:] = 0
        
        if xshift > 0:
            arr[:,:xshift] = 0
        elif xshift < 0:
            arr[:,xshift:] = 0
        

    
    def transform(
        self,
        arr: np.ndarray,
    ):
        results = []
        
        h, w = arr.shape[:2]
        output_h = h - 2 * self.ymax
        output_w = w - 2 * self.xmax
        for xshift in self.xvals():
            for yshift in self.yvals():

                # Append new arr based on mode
                match self.mode:
                    case "crop":
                        x = xshift + self.xmax
                        y = yshift + self.ymax
                        res = arr[
                            y:y + output_h,
                            x:x + output_w
                        ]
                        
                    case "replicate":
                        res = shift(arr, shift=(yshift, xshift, 0), mode="nearest")
                        
                    case "zeros":
                        res = np.roll(arr, shift=(yshift, xshift), axis=(0, 1))
                        Translator.set_borders(res, shift=(xshift, yshift))

                    case "roll":
                        res = np.roll(arr, shift=(yshift, xshift), axis=(0, 1))
                        
                results.append(res)
            
                
        return results

@dataclass
class Rotator(Transformer):
    """
    A class to handle image rotation.
    """
    optype: ClassVar[str] = "rotation"
    rotmax: int = 30
    rotnum: int = 1

    def get_rotvals(self):
        return [
            round(self.rotmax * (rot / self.rotnum)) 
            for rot 
            in range(-self.rotnum, self.rotnum + 1) 
            if rot != 0
        ]
    
    def transform(
        self,
        arr: np.ndarray,
    ):
        h, w = arr.shape[:2]
        center = (w // 2, h // 2)
        results = []

        for rot in self.get_rotvals():
            rotation_matrix = cv2.getRotationMatrix2D(center, rot, 1.0)
            res = cv2.warpAffine(arr, rotation_matrix, (w, h))
            results.append(res)

        return results


@dataclass
class PCA(Transformer):
    """
    A class to handle PCA for images.
    """
    optype: ClassVar[str] = "PCA"
    n_components: float = 3560
    pca = None

    def apply(
        self,
        arr_list: list[np.ndarray],
    ):
        num_arrs = len(arr_list)
        print("Flattening Images...")
        flattened_ims = np.array([arr.flatten() for arr in arr_list])
        self.pca = decomposition.PCA(n_components=self.n_components)
        print("Fitting PCA...")
        self.pca.fit(flattened_ims)
        print("Performing PCA...")
        X = self.pca.transform(flattened_ims)
        return X


@dataclass
class Flatten(Transformer):
    """
    A class to handle image flattening.
    """
    optype: ClassVar[str] = "flattening"

    def apply(
        self,
        arr_list: list[np.ndarray],
    ):
        num_arrs = len(arr_list)
        print(f"Flattening {num_arrs} images...")
        return np.array([arr.flatten() for arr in arr_list])

    def transform(
        self,
        arr: np.ndarray,
    ):
        return arr.flatten()
    

@dataclass
class GaussianFilter(Transformer):
    """
    A class to apply a gaussian filter on an image.
    """
    optype: ClassVar[str] = "gaussian filter"
    size: int
    sigma: float

    def gkern(self, l=5, sig=1.):
        """
        creates gaussian kernel with side length `l` and a sigma of `sig`
        """
        ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
        gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
        kernel = np.outer(gauss, gauss)
        return kernel / np.sum(kernel)

    def transform(
        self,
        arr: np.ndarray,
    ):
        kernel = self.gkern(l=self.size, sig=self.sigma)
        return convolve2d(arr, kernel, mode="valid")


@dataclass
class MedianFilter(Transformer):
    """
    A class to apply a median filter on an image.
    """
    optype: ClassVar[str] = "median filter"
    size: int

    def transform(
        self,
        arr: np.ndarray,
    ):
        assert self.size % 2 == 1, "Median filter must have odd size"
        return medfilt2d(arr, kernel_size=self.size)


@dataclass
class Gradient(Transformer):
    """
    A class to apply a gradient on an image.
    """
    optype: ClassVar[str] = "gradient"

    def transform(
        self,
        arr: np.ndarray,
    ):
        dx_kernel = np.array([[-1, 1]])
        dy_kernel = np.array([[-1],
                              [1]])
        dx = convolve2d(arr, dx_kernel, mode="valid")[:-1]
        dy = convolve2d(arr, dy_kernel, mode="valid")[:,:-1]

        return np.sqrt(dx**2 + dy**2)
        

@dataclass
class Threshold(Transformer):
    """
    A class to apply a threshold on an image.
    """
    optype: ClassVar[str] = "thresholding"
    threshold: float

    def transform(
        self,
        arr: np.ndarray,
    ):
        arr[arr > self.threshold] = 255
        arr[arr <= self.threshold] = 0

        return arr