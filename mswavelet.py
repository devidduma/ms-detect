import os
import shutil
import PIL.Image
import imageio
from mat4py import loadmat
import PIL
import matplotlib.pyplot as plt
import numpy as np
import pywt
import torch


class MSWavelet:
    def __init__(self, rootdir="./MRIFreeDataset", outputdir="./Preprocessed"):
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
            point_index = plaque.rfind(".")

            try:
                candidate2 = self.tif_filename(plaque[:underscore_index])
                if candidate2 not in self.unhealthy_images:
                    self.unhealthy_images.append(candidate2)
            except Exception as e:
                pass

            try:
                candidate1 = self.tif_filename(plaque[:point_index])
                if candidate1 not in self.unhealthy_images:
                    self.unhealthy_images.append(candidate1)
            except Exception as e:
                pass

    def pop_healthy_images(self):
        for img in self.imagearray:
            point_index = img.rfind(".")

            try:
                candidate = self.tif_filename(img[:point_index])
                if candidate not in self.unhealthy_images:
                    self.healthy_images.append(candidate)
            except Exception as e:
                pass

    def head_path_depth(self, path, depth=1):
        tail = ""

        for d in range(depth):
            tail = os.path.split(path)[1]
            path = os.path.split(path)[0]

        return path, tail

    def output_path(self, imgpath):
        imgpath = imgpath[len(self.rootdir) + 1:]
        imgpath = os.path.join(self.outputdir, imgpath)

        return imgpath

    def create_directories_output(self):
        if not os.path.exists(self.outputdir):
            os.makedirs(self.outputdir)

        for imgpath in self.imagearray:
            imgpath = self.output_path(imgpath)

            os.makedirs(self.head_path_depth(imgpath, depth=3)[0], exist_ok=True)
            os.makedirs(self.head_path_depth(imgpath, depth=2)[0], exist_ok=True)
            os.makedirs(self.head_path_depth(imgpath, depth=1)[0], exist_ok=True)

        for plaquepath in self.plaquearray:
            outputpath = self.output_path(plaquepath)

            shutil.copy(plaquepath, outputpath)

    def filename_stripped(self, filename):
        filename = str(filename)

        if filename.lower().endswith(".tif") or filename.lower().endswith(".bmp"):
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

    # Continuous Wavelet
    def plot_continuous_wavelet(self, wavelettype, filename=None, file=None, scales=np.arange(1, 8, 2), cmap=None):
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
    def upsample(self, files, size, mode="bilinear", info=True):
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
        newimg = newimg.reshape(newimg.shape[4 - ndims:])
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

    # generate training samples
    def Xprocess(self, file, cwtlist, cwtscales, attach_original):
        # cast files to numpy array
        file = np.asarray(file)

        # List of results of Wavelet Transforms
        reswt = []

        if attach_original:
            attach = np.expand_dims(file, axis=0)
            attach = np.asarray(attach, dtype=self.WT_DTYPE)
            reswt.append(attach)

        # Continuous wavelet transforms
        for i in range(len(cwtlist)):
            res, _ = pywt.cwt(file, wavelet=cwtlist[i], scales=cwtscales[i])
            res = np.asarray(res, dtype=self.WT_DTYPE)
            # append only last WT
            reswt.append(res)

        # Concatenate channels from wavelet transforms
        # add wavelet transforms
        res = np.asarray(np.concatenate(reswt, axis=0), dtype=self.WT_DTYPE)

        return res

    def batch_Xproc_outputdir(self, cwtlist, cwtscales, size, attach_original):
        print("Starting batch preprocessing phase...")

        self.create_directories_output()

        for imgname in self.imagearray:
            imgfile = PIL.Image.open(self.tif_filename(imgname))

            # in case it is a 512x512 file, clean the sides
            if np.asarray(imgfile).shape[-1] == 512:
                imgfile = self.clean_images512_sides(imgfile)

            imgups = self.upsample(imgfile, size=size, info=False)

            output = self.Xprocess(imgups, cwtlist, cwtscales, attach_original)
            output = np.transpose(output, axes=(1, 2, 0))
            output = np.asarray(output, dtype=np.uint8)

            imgname, _ = os.path.splitext(imgname)
            final_imgname = imgname + ".tif"
            path_to_save = self.output_path(final_imgname)
            imageio.imwrite(path_to_save, output)

        print("Batch preprocessing finished!")

    def clean_images512_sides(self, img):
        img = np.array(img)
        if img.shape[-1] != 512:
            raise Exception("It is meant to clean only 512x512 images!")

        fillvalue = 0
        topleft, bottomright = (70, 50), (70, 50)
        bottomleft = (65, 65)
        right = (35, 167, 190)

        for icol in range(topleft[0]):
            for irow in range(topleft[1]):
                img[irow, icol] = fillvalue

        for icol in range(bottomright[0]):
            for irow in range(bottomright[1]):
                img[img.shape[0] - 1 - irow, img.shape[1] - 1 - icol] = fillvalue

        for icol in range(bottomleft[0]):
            for irow in range(bottomleft[1]):
                img[img.shape[0] - 1 - irow, icol] = fillvalue

        for icol in range(right[0]):
            for irow in range(right[1], right[2]):
                img[irow, img.shape[1] - 1 - icol] = fillvalue

        return img


if __name__ == '__main__':
    # Demo
    msw = MSWavelet(outputdir="./PreprocessedV2")
    msw.printinfo()

    """
    imgname = msw.unhealthy_images[110]
    img = PIL.Image.open(imgname)
    imgclean = msw.clean_images512_sides(img)
    plt.imshow(X=img, interpolation="nearest")
    plt.show()
    plt.imshow(X=imgclean, interpolation="nearest")
    plt.show()
    """

    cwtlist = ["mexh", "fbsp"]
    cwtscales = [12, 16]
    dwtlist = ["haar"]

    # Batch continuous wavelet transform and save to disk
    msw.batch_Xproc_outputdir(cwtlist, cwtscales, size=512, attach_original=True)

    """
    # Plot unhealthy image + wavelet transforms
    cwtscales_plot = [np.arange(5, 20, 4), np.arange(5, 20, 4)]
    msw.plot_transforms(wtlist=cwtlist, type="continuous", cwtscales=cwtscales_plot, imgname=msw.unhealthy_images[100])
    msw.plot_transforms(wtlist=dwtlist, type="discrete", imgname=msw.unhealthy_images[100])
    """
