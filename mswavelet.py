import os
import shutil

import PIL.Image
from mat4py import loadmat
import PIL
import matplotlib.pyplot as plt
from sklearn import covariance
import numpy as np
import pywt
import torch


class MSWavelet:
    def __init__(self, rootdir="./MRIFreeDataset", outputdir="./Collected"):
        # Rootdir path
        self.rootdir = rootdir
        self.outputdir = outputdir

        # Collect the MRI data
        self.imagearray = []
        self.plaquearray = []
        self.unhealthy_images = []
        self.healthy_images = []

        # Plotting size (x100)
        self.figwidth = 16
        self.figheight = 8

        # dtype
        self.WT_DTYPE = np.float16

        # init
        self.pop_imagearray_plaquearray()
        self.pop_unhealthy_images()
        self.pop_healthy_images()

        # clean
        self.imagearray = self.__clean_array__(self.imagearray)
        self.unhealthy_images = self.__clean_array__(self.unhealthy_images)
        self.healthy_images = self.__clean_array__(self.healthy_images)

        print("MSWavelet arrays: ")
        print("Number of MRI scans: ", len(self.imagearray))
        print("Number of unhealthy MRI scans: ", len(self.unhealthy_images))
        print("Number of healthy MRI scans: ", len(self.healthy_images))
        print("")

    def __clean_array__(self, array):
        result = []
        for fname in array:
            try:
                self.tif_filename(filename=fname)
            except Exception as e:
                continue
            result.append(fname)
        return result

    # Collect data for image and plaque files
    def pop_imagearray_plaquearray(self):
        for subdir, dirs, files in os.walk(self.rootdir):
            for file in files:
                if file.lower().endswith(".tif") or file.lower().endswith(".bmp"):
                    self.imagearray.append(os.path.join(subdir, file))
                elif file.lower().endswith(".plq"):
                    self.plaquearray.append(os.path.join(subdir, file))
                # print(os.path.join(subdir, file))

    # Select only unhealthy images in a separate array
    def pop_unhealthy_images(self):
        for plaque in self.plaquearray:
            underscore_index = plaque.rfind("_")
            im_index = plaque.lower().rfind("im")
            point_index = plaque.rfind(".")

            if plaque[im_index:underscore_index].lower() == "im":
                self.unhealthy_images.append(self.tif_filename(plaque[:point_index]))
            elif self.tif_filename(plaque[:underscore_index]) not in self.unhealthy_images:
                self.unhealthy_images.append(self.tif_filename(plaque[:underscore_index]))

    def pop_healthy_images(self):
        for img in self.imagearray:
            point_index = img.rfind(".")
            if self.tif_filename(img[:point_index]) not in self.unhealthy_images:
                self.healthy_images.append(self.tif_filename(img[:point_index]))

    def create_directories(self):
        if not os.path.exists(self.outputdir):
            os.makedirs(self.outputdir)

        counter = 1
        for src_path in self.unhealthy_images:
            dst_folder = os.path.join(self.outputdir, "1")
            if not os.path.exists(dst_folder):
                os.makedirs(dst_folder)

            basename = os.path.basename(self.tif_filename(src_path))
            ext = os.path.splitext(basename)[1]
            stripped = self.filename_stripped(basename)
            stripped += "_" + str(counter)
            basename = stripped + ext
            counter += 1

            dst = os.path.join(dst_folder, basename)
            shutil.copyfile(self.tif_filename(src_path), dst)

        for src_path in self.healthy_images:
            dst_folder = os.path.join(self.outputdir, "0")
            if not os.path.exists(dst_folder):
                os.makedirs(dst_folder)

            basename = os.path.basename(self.tif_filename(src_path))
            ext = os.path.splitext(basename)[1]
            stripped = self.filename_stripped(basename)
            stripped += "_" + str(counter)
            basename = stripped + ext
            counter += 1

            dst = os.path.join(dst_folder, basename)
            shutil.copyfile(self.tif_filename(src_path), dst)

    def filename_stripped(self, filename):
        filename = str(filename)

        if filename.lower().endswith(".tif") or filename.lower().endswith(".bmp") or filename.lower().endswith(".tiff"):
            point_index = filename.rfind(".")
            filename = filename[:point_index]

        return filename

    # Return image file name
    def tif_filename(self, filename):
        filename = self.filename_stripped(filename)

        if os.path.exists(filename + ".bmp"):
            filename += ".bmp"
        elif os.path.exists(filename + ".BMP"):
            filename += ".BMP"
        elif os.path.exists(filename + ".tif"):
            filename += ".tif"
        elif os.path.exists(filename + ".TIF"):
            filename += ".TIF"
        elif os.path.exists(filename + ".tiff"):
            filename += ".tiff"
        elif os.path.exists(filename + ".TIFF"):
            filename += ".TIFF"
        else:
            raise Exception("Filename not supported!", filename)

        return filename

    # Return plaque file name extension
    def mat_filename(self, filename, index=0):
        # strip filename
        filename = self.filename_stripped(filename)

        if index == 0:
            filename = filename
        else:
            filename = filename + "_" + str(index)

        # end with extension .plq
        filename = filename + ".plq"
        return filename

    # Return plaque file
    def mat(self, filename, index=0):
        # find filename for plaque with index
        filename = self.mat_filename(filename, index=index)

        # find plaque file
        file = None
        if os.path.exists(filename):
            file = loadmat(filename)
        return file

    # List all plaque files for image
    def list_file_lesions(self, filename):
        # strip filename
        filename = self.filename_stripped(filename)

        # list plaque files
        result = []
        counter = 0
        while True:
            file = self.mat(filename, index=counter)
            if counter == 0 and file is None:
                counter = 1
                continue
            if file is None:
                break
            result.append(self.mat_filename(filename, index=counter))
            counter += 1

        return result

    # Centroid of lesion
    def centroid(self, xi, yi):
        xi = np.asarray(xi)
        yi = np.asarray(yi)

        xmean = np.mean(xi, axis=0)
        ymean = np.mean(yi, axis=0)

        return xmean, ymean

    def removeticks(self, ax):
        ax.tick_params(
            axis='both',  # changes apply to both axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            left=False,  # ticks along the top edge are off
            labelbottom=False,  # labels along the bottom edge are off
            labelleft=False  # labels along the left edge are off
        )

    # .plq files contain spatial coordinates for lesions in "xi" and "yi" key
    def plot_lesions(self, filename, file=None, ax=plt, noticks=True, cmap=None):
        if file is not None:
            img = file
        else:
            img = PIL.Image.open(self.tif_filename(filename))

        # Plot image
        ax.imshow(X=img, cmap=cmap)

        # remove ticks
        if noticks:
            self.removeticks(ax)

        # Plot lesions
        for flname in self.list_file_lesions(filename):
            filelesion = loadmat(flname)

            xi = filelesion["xi"]
            yi = filelesion["yi"]
            originalimg = PIL.Image.open(self.tif_filename(filename))
            xi, yi = self.linear_fit_range(xi, yi, np.asarray(originalimg).shape, np.asarray(img).shape)

            ax.plot(xi, yi, color="black")
            """
            xmean, ymean = self.centroid(filename)
            plt.scatter(xmean, ymean, c="black")
            """

        if ax == plt:
            plt.show()

    def img_with_lesions(self, file, filename, original_size, dtype, size, color="black"):
        if np.asarray(file).ndim != 2:
            raise Exception("Only files with 2 dimensions are allowed for this method!")

        pass

    # Learn elliptic envelope for lesion
    def fit_elliptic_curve(self, filename, index = 0, plot = True):
        file = self.mat(filename, index)
        x = np.concatenate((file["xi"], file["yi"]), axis=1)
        # print(x.shape)

        model = covariance.EllipticEnvelope()
        scores = model.fit_predict(x)

        if plot:
            anom_index = np.where(scores==-1)
            values = x[anom_index]

            plt.scatter(x[:,0], x[:,1])
            plt.scatter(values[:,0],values[:,1], color='r')
            plt.show()

        return model

    # Continuous Wavelet
    def plot_continuous_wavelet(self, wavelettype, filename=None, file=None, scales=np.arange(1,8,2), cmap=None):
        # Load image
        # original = pywt.data.camera()
        if file is not None:
            img = file
        elif filename is not None:
            img = PIL.Image.open(self.tif_filename(filename))
        else:
            raise Exception("Either filename or file must be specified.")

        # Wavelet transform of image, and plot approximation and details
        titles = ['Scale ' + str(scales[0]), 'Scale ' + str(scales[1]),
                  'Scale ' + str(scales[2]), 'Scale ' + str(scales[3])]

        # coeffs2 = pywt.dwt2(original, 'bior1.3')
        coeffs2, freq = pywt.cwt(img, scales=scales, wavelet=wavelettype)
        coeffs2 = np.array(coeffs2, dtype=self.WT_DTYPE)
        LL, LH, HL, HH = coeffs2

        self.__plot_lesions_for_wt__(filename, [LL, LH, HL, HH], titles, cmap=cmap)

        return coeffs2

    # Discrete Wavelet 2D
    def plot_discrete_wavelet_2d(self, wavelettype, filename=None, file=None, cmap=None):
        # Load image
        # original = pywt.data.camera()
        if file is not None:
            img = file
        elif filename is not None:
            img = PIL.Image.open(self.tif_filename(filename))
        else:
            raise Exception("Either filename or file must be specified.")

        # Wavelet transform of image, and plot approximation and details
        titles = ['Approximation', ' Horizontal detail',
                  'Vertical detail', 'Diagonal detail']

        LL, coeffs2 = pywt.dwt2(img, wavelet=wavelettype)
        coeffs2 = np.array(coeffs2, dtype=self.WT_DTYPE)
        LH, HL, HH = coeffs2

        self.__plot_lesions_for_wt__(filename, [LL, LH, HL, HH], titles, cmap=cmap)

        return coeffs2

    def __plot_lesions_for_wt__(self, filename, tuple, titles, cmap=None):
        # list of plaque files
        filelesionslist = self.list_file_lesions(filename)
        print("Plaque files for image: ", filelesionslist)

        # Cast for plotting
        tuple = np.array(tuple, dtype=np.float64)

        LL, LH, HL, HH = tuple

        num_rasters = 1
        if len(filelesionslist) != 0:
            num_rasters = 2
        fig, ax = plt.subplots(num_rasters, 4)

        for i, a in enumerate([LL, LH, HL, HH]):
            ax0i = ax[0, i] if num_rasters == 2 else ax[i]

            self.removeticks(ax0i)
            ax0i.imshow(X=a, interpolation="nearest", cmap=cmap)
            ax0i.set_title(titles[i], fontsize=10)
            if num_rasters == 2:
                self.plot_lesions(filename, file=a, ax=ax[1, i], noticks=True, cmap=cmap)

        fig.set_figwidth(self.figwidth)
        fig.set_figheight(self.figheight)
        fig.tight_layout()
        plt.show()

    # Rescale lesion position
    def linear_fit_range(self, xi, yi, imgshape, ashape):
        xi = np.asarray(xi) / imgshape[0] * ashape[0]
        yi = np.asarray(yi) / imgshape[1] * ashape[1]

        return xi, yi

    # Supports batch upsampling, which is very fast
    def upsample(self, files, size, mode="nearest", info=True):
        # torchlayer = torch.nn.UpsamplingNearest2d(size=(size, size))

        # cast to numpy array only if not already numpy array
        if not isinstance(files, np.ndarray):
            filenp = np.array(files)
        else:
            filenp = files

        ndims = None
        if filenp.ndim == 2:
            ndims = 2
            filenp = np.expand_dims(filenp, axis=0)
            filenp = np.expand_dims(filenp, axis=0)
        elif filenp.ndim == 3:
            ndims = 3
            filenp = np.expand_dims(filenp, axis=0)
        elif filenp.ndim == 4:
            ndims = 4
        else:
            raise Exception("Number of dimensions of tensor must be 2, 3 or 4, but shape ", filenp.shape, " found.")

        originaltype = type(filenp[0, 0, 0, 0])

        # Return if size already satisfied
        currsize = np.asarray(files).shape[-2:]
        if currsize == (size, size):
            return np.array(files, dtype=originaltype)

        # Cast accordingly
        if originaltype in [np.float16, np.int8, np.uint8, np.int16, np.uint16]:
            filenp = np.array(filenp, dtype=float)
        else:
            filenp = np.array(filenp, dtype=np.float64)

        # Upsample
        filetorch = torch.tensor(filenp)
        newimg = torch.nn.functional.interpolate(filetorch, size=(size, size), mode=mode)
        newimg = np.array(newimg, dtype=originaltype)

        # Return with original shape and type
        newimg = newimg.reshape(newimg.shape[4-ndims:])
        newimg = np.array(newimg, dtype=originaltype)

        if info:
            print("Upsampled shape: ", newimg.shape, "Original type: ", originaltype)

        return newimg

    def printinfo(self):
        # Print list of discrete and continuous wavelets
        wavelist = pywt.wavelist(kind='discrete')
        print("Discrete Wavelets list: ", wavelist)
        wavelist = pywt.wavelist(kind='continuous')
        print("Continuous Wavelets list: ", wavelist)
        print("")

    def plot_transforms(self, wtlist, type, imgname, cwtscales=None):
        if type not in ["continuous", "discrete"]:
            raise Exception("Type of wavelet transform not recognized.")

        print("\nUPSAMPLED IMAGE:\n")
        imgfile = PIL.Image.open(self.tif_filename(imgname))
        imgups = self.upsample(imgfile, size=1024)
        self.plot_lesions(imgname, file=imgups)
        print("Newimg: ", imgups.shape)

        for i in range(len(wtlist)):
            print("\n", wtlist[i].upper(), type.upper(), "TRANSFORM\n")
            if type == "continuous":
                self.plot_continuous_wavelet(wtlist[i], imgname, scales=cwtscales[i], file=imgups)
            elif type == "discrete":
                self.plot_discrete_wavelet_2d(wtlist[i], imgname, file=imgups)


if __name__ == '__main__':
    # Demo
    msw = MSWavelet()
    msw.printinfo()

    msw.create_directories()

    """
    cwtlist = ["gaus4", "fbsp", "cmor", "shan", "cgau4"]
    cwtscales = [np.arange(1,23,7)]*5
    msw.plot_transforms(wtlist=cwtlist, type="continuous", cwtscales=cwtscales, imgname=msw.unhealthy_images[100])
    """
    """
    # Batch Upsampling
    print("Batch Upsampling:")
    imgnames = msw.unhealthy_images[:20]
    imgfiles = []
    for imgname in imgnames:
        imgfile = PIL.Image.open(msw.tif_filename(imgname))
        imgfilenumpy = np.asarray(imgfile)
        if imgfilenumpy.shape[0] == 512:
            imgfiles.append(imgfilenumpy)
    imgups = msw.upsample(imgfiles, size=1024)
    print("Upscale images array: ", imgups.shape)

    # Wavelet Transforms
    dwtlist = ["bior6.8", "haar", "rbio4.4", "sym12"]
    cwtlist = ["mexh", "morl", "gaus4"]
    cwtscales = [np.arange(1,8,2), np.arange(1,20,6), np.arange(1,14,4)]

    # Batch Wavelet Transforms
    img, coeffs_dwt2 = pywt.dwt2(imgups, wavelet=dwtlist[0])
    print("Batch DWT2D array shape: ", np.asarray(coeffs_dwt2).shape)
    coeffs_cwt, freqs = pywt.cwt(imgups, wavelet=cwtlist[0], scales=np.arange(1,8,2))
    print("Batch CWT array shape: ", np.asarray(coeffs_cwt).shape)
    coeffs_dwt2, coeffs_cwt = [], []    # free memory

    # Plot unhealthy image + wavelet transforms
    msw.plot_transforms(wtlist=cwtlist, type="continuous", cwtscales=cwtscales, imgname=msw.unhealthy_images[100])
    msw.plot_transforms(wtlist=dwtlist, type="discrete", imgname=msw.unhealthy_images[100])
    """""