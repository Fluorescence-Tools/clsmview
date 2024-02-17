"""
CLSM pixel select
is a program that can be used to create representations of CLSM TTTR data,
to select pixels, and create and export fluorescence decay histograms. Exported
decay histograms can be directly used in ChiSurf.
"""
from __future__ import annotations

import typing

import sys
import os
import yaml
import copy
import pathlib

import pyqtgraph
import pyqtgraph.dockarea
import pyqtgraph.graphicsItems.GradientEditorItem
from qtpy import QtCore, QtGui, QtWidgets, uic

import numpy as np
import numba as nb
import pandas as pd

import skimage as ski
import scipy.stats as st

import tttrlib


class DataCurve(object):

    @property
    def x(self):
        return self._df['x']

    @x.setter
    def x(self, v):
        self._df['x'] = v
    @property
    def y(self):
        return self._df['y']

    @y.setter
    def y(self, v):
        self._df['y'] = v

    def __init__(self, x, y, ey, **kwargs):
        self._kwargs = kwargs
        self._df = pd.DataFrame(np.vstack([x, y, ey]).T, columns=['x', 'y', 'ey'])

"""
# try:
import chisurf

import chisurf.gui
import chisurf.data
from chisurf.data import DataCurve

import chisurf.settings
import chisurf.experiments
# from chisurf.gui.widgets.experiments import DataCurve
plot_settings = chisurf.settings.gui['plot']
imported_datasets: typing.List[chisurf.data.DataCurve] = chisurf.imported_datasets
# except ImportError:
#     chisurf = None
#     from . selector import ExperimentalDataSelector
#     with open(pathlib.Path(__file__).parent / 'settings.yaml') as fp:
#         settings = yaml.safe_load(fp)
#         plot_settings = settings['gui']['plot']
#     imported_datasets = list()
"""

try:
    import chisurf
    import chisurf.settings
    from chisurf.gui.widgets.experiments import ExperimentalDataSelector
    plot_settings = chisurf.settings.gui['plot']
    imported_datasets = chisurf.imported_datasets
    from chisurf.data import DataCurve
    colors = chisurf.settings.colors
except ModuleNotFoundError:
    from clsmview.selector import ExperimentalDataSelector
    with open(pathlib.Path(__file__).parent / 'settings.yaml') as fp:
        settings = yaml.safe_load(fp)
        plot_settings = settings['gui']['plot']
    imported_datasets = list()
    chisurf = None
    experiment = None
    from . colors import colors


clsm_settings = yaml.safe_load(
    open(
        pathlib.Path(__file__).parent / pathlib.Path("clsm_settings.yaml")
    )
)



def counting_noise(
        decay: np.ndarray,
        treat_zeros: bool = True,
        zero_value: float = 1.0
) -> np.array:
    """Calculated Poisson noise (sqrt of counts) for TCSPC fluorescence decays

    Parameters
    ----------
    decay : numpy-array
        the photon counts
    treat_zeros: bool
        If treat zeros is True (default) a number specified by `zero_value`
        will be assigned for cases the decay is zero.
    zero_value : float
        The number that will be assigned cases where the value of the decay is
        zero.

    Returns
    -------
    numpy-array
        an array containing noise estimates of the decay that can ve used in
        the data analysis.

    """
    w = np.array(decay, dtype=np.float64)
    if treat_zeros:
        w[w <= 0.0] = zero_value
    return np.sqrt(w)


def gaussian_kernel(
        kernel_size: int = 21,
        nsig: float = 3
):
    """Returns a 2D Gaussian kernel array.

    :param kernel_size: the size of the 2D array
    :param nsig: the width of the gaussian in the 2D array.
    :return:
    """
    interval = (2.0 * nsig + 1.) / kernel_size
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernel_size+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel



@nb.jit(nopython=True)
def _frc_histogram(lx, rx, ly, ry, f1f2, f12, f22, n_bins, bin_width):
    """Auxiliary function only intented to be used by compute_frc

    Parameters
    ----------
    lx : int
        left boundary of the x axis. For an image with 512 pixel this
        would be -256
    rx : int
        right boundary of the x axis. For 512 pixel 255
    ly : int
        left boundary of the x axis. For an image with 512 pixel this
        would be -256
    ry : int
        right boundary of the x axis. For 512 pixel 255
    f1f2 : numpy.array
        The product of the Fourier transformed images F(1)F(2)',
        where F(2)' is the complex conjugate of F2
    f12 : numpy.array
        The squared absolute value of the F(1) Fourier transform, abs(F(1))**2
    f22 : numpy.array
        The squared absolute value of the F(2) Fourier transform, abs(F(2))**2
    n_bins : int
        The number of bins in the FRC
    bin_width : float
        The width of the Rings in the FRC

    Returns
    -------
    numpy-array:
        The FRC value

    """
    wf1f2 = np.zeros(n_bins, np.float64)
    wf1 = np.zeros(n_bins, np.float64)
    wf2 = np.zeros(n_bins, np.float64)
    for xi in range(lx, rx):
        for yi in range(ly, ry):
            distance_bin = int(np.sqrt(xi ** 2 + yi ** 2) / bin_width)
            if distance_bin < n_bins:
                wf1f2[distance_bin] += f1f2[xi, yi]
                wf1[distance_bin] += f12[xi, yi]
                wf2[distance_bin] += f22[xi, yi]
    return wf1f2 / np.sqrt(wf1 * wf2)



def compute_frc(
        image_1: np.ndarray,
        image_2: np.ndarray,
        bin_width: int = 2.0
):
    """Computes the Fourier Ring Correlation (FRC) between two images.

    The FRC measures the correlation between two images by the overlap of the
    Fourier transforms. The FRC is the normalised cross-correlation coefficient
    between two images over corresponding shells in Fourier space transform.

    Parameters
    ----------
    image_1 : numpy.array
        The first image
    image_2 : numpy.array
        The second image
    bin_width : float
        The bin width used in the computation of the FRC histogram

    Returns
    -------
    Numpy array:
        density of the FRC histogram
    Numpy array:
        bins of the FRC histogram

    """
    f1 = np.fft.fft2(image_1)
    f2 = np.fft.fft2(image_2)
    f1f2 = np.real(f1 * np.conjugate(f2))
    f12, f22 = np.abs(f1) ** 2, np.abs(f2) ** 2
    nx, ny = image_1.shape

    bins = np.arange(0, np.sqrt((nx // 2) ** 2 + (ny // 2) ** 2), bin_width)
    n_bins = int(bins.shape[0])
    lx, rx = int(-(nx // 2)), int(nx // 2)
    ly, ry = int(-(ny // 2)), int(ny // 2)
    density = _frc_histogram(
        lx, rx,
        ly, ry,
        f1f2, f12, f22,
        n_bins, bin_width
    )
    return density, bins



class CLSMPixelSelect(QtWidgets.QWidget):
    """
    Attributes
    ----------
    current_image : numpy.array
        The currently displayed image
    current_image_subset_1 : numpy.array
        A subset of frames different from the current image that is used to
        estimate the resolution in the FRC. If the sum or the mean of the frames
        is displayed the images that are averages are split in two sets that are
        used to compute the FRC. Only the current image is displayed. The FRC is
        computed using the neighboring images (for the first frame, the first frames
        is current_image_subset_1)
    current_image_subset_2 : numpy.array
        A subset of frames different from the current image that is used to
        estimate the resolution in the FRC
    tttr_data : tttrlib.TTTR
        The TTTR dataset used to generate the image
    clsm_images : dict
        A dictionary that contains the generated tttrlib.CLSMImage objects
    clsm_representations : dict
        A clsm image can have multiple representation, e.g., intensity images
        mean arrival time images, etc. The clsm_representations dictionary stores
        the different representations for clsm images
    brush_kernel : numpy.ndarray
        The kernel that is used to brush a selection. The brush kernel is a
        Gaussian with a width defined by brush_width
    brush_size : int
        The size of the brush kernel
    brush_width : float
        Defines the v
    rois : dict
        The user can define a set of rois. The rois dictionary stores these
        user defined rois

    """

    tttr_data: tttrlib.TTTR = None
    current_image: np.ndarray = None
    current_image_subset_1: np.ndarray = None
    current_image_subset_2: np.ndarray = None

    clsm_images: typing.Dict[str, tttrlib.CLSMImage] = dict()
    clsm_representations: typing.Dict[str, np.ndarray] = dict()

    brush_kernel: np.ndarray = None
    brush_size: int = 7
    brush_width: float = 3
    rois: typing.Dict[str, np.ndarray] = dict()

    img_plot: pyqtgraph.PlotWidget = None
    frc_plot_window: pyqtgraph.PlotWidget = None
    current_decay = DataCurve([], [], [])

    def get_data_curves(self, *args, **kwargs) -> typing.List[DataCurve]:
        return self._curves

    @property
    def curve_name(self):
        s = str(self.lineEdit.text())
        if len(s) == 0:
            return "no-name"
        else:
            return s

    @property
    def filename(self):
        return self.lineEdit.text()

    @filename.setter
    def filename(self, v: pathlib.Path):
        self.lineEdit.setText(str(v.as_posix()))

    @property
    def n_ph_min(self):
        return int(self.spinBox.value())

    @n_ph_min.setter
    def n_ph_min(self, v: int):
        self.spinBox.setValue(v)

    @property
    def stack_frames(self):
        return self.radioButton_4.isChecked() or self.radioButton_5.isChecked()

    @property
    def tac_coarsening(self):
        return int(self.comboBox_6.currentText())

    @property
    def gradient_key(self):
        return self.comboBox_2.currentText()

    @gradient_key.setter
    def gradient_key(self, v: typing.Tuple[int, typing.List[str]]):
        idx, g_keys = v
        if len(g_keys) > 0:
            self.comboBox_2.addItems(g_keys)
        self.comboBox_2.setCurrentIndex(idx)

    @property
    def current_image_type(self):
        return str(self.comboBox_3.currentText())

    @property
    def current_clsm_name(self):
        return str(self.comboBox_9.currentText())

    @property
    def current_image_name(self):
        return self.current_clsm_name + "_" + self.current_image_type

    @property
    def current_roi_name(self):
        try:
            name = self.listWidget.currentItem().text()
        except AttributeError:
            name = self.lineEdit_5.text()
        if len(name) == 0:
            name = 'no-name'
        return name

    @property
    def current_frame(self):
        return self.spinBox_8.value()

    def set_current_frame_range(self, fmin, fmax):
        self.spinBox_8.setMinimum(fmin)
        self.spinBox_8.setMaximum(fmax)

    @property
    def do_live_updates(self):
        return self.checkBox_4.isChecked()

    @property
    def plot_sum_of_frames(self):
        return self.radioButton_4.isChecked()

    @property
    def plot_mean_of_frames(self):
        return self.radioButton_5.isChecked()

    def onRemoveDataset(self):
        selected_index = [
            i.row() for i in self.cs.selectedIndexes()
        ]
        l = list()
        for i, c in enumerate(self._curves):
            if i not in selected_index:
                l.append(c)
        self._curves = l
        self.cs.update()
        self.plot_all_curves()

    def update_selection_plot(self):
        plot = self.plot.getPlotItem()
        curve = plot.curves[0]
        curve.setData(
            x=self.current_decay.x,
            y=self.current_decay.y
        )

    def plot_all_curves(self) -> None:
        plot = self.plot.getPlotItem()
        plot.clear()
        curve = self.current_decay
        plot.plot(curve.x, curve.y, name="Current selection")

        self.legend = plot.addLegend()
        # plot.setLogMode(x=False, y=True)
        plot.showGrid(True, True, 1.0)
        plot.setLogMode(False, True)

        current_curve = self.cs.selected_curve_index
        lw = plot_settings['line_width']
        for i, curve in enumerate(self._curves):
            w = lw * 0.5 if i != current_curve else 2.5 * lw
            color = colors[i % len(colors)]['hex']
            pen = pyqtgraph.mkPen(color, width=w)
            plot.plot(curve.x, curve.y, pen=pen, name=curve.name)
        plot.autoRange()

    def add_curve(self, *args, v=None, **kwargs):
        decay = copy.copy(self.current_decay)
        roi_name = self.current_roi_name
        decay.name = self.current_clsm_name + "_ROI(" + roi_name + ")"
        self._curves.append(decay)
        # Update UI
        self.cs.update()
        self.plot_all_curves()

    def open_tttr_file(self, *args, filename: str = None, tttr_type: str = None) -> None:
        # Load from UI
        if not isinstance(filename, str):
            tentative_filename = str(self.lineEdit.text())
            if os.path.isfile(tentative_filename):
                path = pathlib.Path(tentative_filename).parent
            elif chisurf is not None:
                path = chisurf.working_path
            else:
                path = pathlib.Path.home()
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "TTTR CLSM file", str(path.as_posix()),
                "*.ht3;*.ptu;*.spc"
            )
        if tttr_type is None:
            tttr_type = str(self.comboBox_4.currentText())
        if os.path.isfile(filename):
            self.filename = pathlib.Path(filename)
            self.tttr_data = tttrlib.TTTR(filename, tttr_type)
            if tttr_type in ('PTU', 'HT3'):
                new_clsm_settings = tttrlib.CLSMImage.read_clsm_settings(self.tttr_data)
                current_setup = self.comboBox_5.currentText()
                clsm_settings[current_setup]['frame_marker'] = new_clsm_settings['marker_frame_start']
                clsm_settings[current_setup]['line_start_marker'] = new_clsm_settings['marker_line_start']
                clsm_settings[current_setup]['line_stop_marker'] = new_clsm_settings['marker_line_stop']
                clsm_settings[current_setup]['event_type_marker'] = new_clsm_settings['marker_event_type']
                clsm_settings[current_setup]['pixel_per_line'] = new_clsm_settings['n_pixel_per_line']
                self.update_clsm_settings(current_setup, clsm_settings)

    def add_representation(
            self,
            *args,
            clsm_image: tttrlib.CLSMImage = None,
            image_type: str = None,
            clsm_name: str = None,
            n_ph_min: int = None,
            image_name: str = None
    ) -> None:
        # Read from UI
        if not isinstance(clsm_name, str):
            clsm_name = self.current_clsm_name
        if not isinstance(clsm_name, np.ndarray):
            clsm_image = self.clsm_images[clsm_name]
        if not isinstance(image_type, str):
            image_type = self.current_image_type
        if not isinstance(n_ph_min, int):
            n_ph_min = self.n_ph_min
        if not isinstance(image_name, str):
            image_name = self.current_image_name
        if image_type == "Mean micro time":
            n_ph_min = self.n_ph_min
            mean_micro_time = clsm_image.get_mean_micro_time(self.tttr_data, n_ph_min, False)
            data = mean_micro_time.astype(np.float64)
        elif image_type == "Intensity":
            data = clsm_image.intensity.astype(np.float64)
        else:
            mean_micro_time = clsm_image.get_mean_micro_time(self.tttr_data, n_ph_min, False)
            intensity_image = clsm_image.intensity
            data = mean_micro_time * intensity_image
        self.clsm_representations[image_name] = data

        # update UI
        self.comboBox_7.clear()
        self.comboBox_7.addItems(
            list(self.clsm_representations.keys())
        )
        self.image_changed()

    def add_roi(self, *args, roi_name: str = None) -> None:
        if roi_name is None:
            roi_name = self.lineEdit_5.text()
        self.rois[roi_name] = self.pixel_selection.image
        # Update UI
        self.listWidget.clear()
        for mn in self.rois.keys():
            self.listWidget.addItem(mn)

    def roi_changed(self, *args) -> None:
        roi_name = self.listWidget.currentItem().text()
        d = self.rois[roi_name]
        d *= np.max(self.img.image.flatten()).astype(np.float64)
        self.pixel_selection.setImage(d)
        self.pixel_selection.updateImage()
        self.update_plot()

    def get_selected_roi(self) -> typing.List[str]:
        # Get list of roi_names from UI
        roi_names = list()
        list_items = self.listWidget.selectedItems()
        if not list_items:
            return list()
        for item in list_items:
            lw = self.listWidget.takeItem(self.listWidget.row(item))
            roi_names.append(lw.text())
        return roi_names

    def remove_rois(self, *args, roi_names: typing.List[str] = None) -> None:
        # Read from UI
        if roi_names is None:
            roi_names = self.get_selected_roi()
        # Remove rois
        for roi_name in roi_names:
            self.rois.pop(roi_name, None)

    def save_pixel_roi(self, event, filename: str = None) -> None:
        if filename is None:
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                None, 'Image file', None, 'All files (*.*)'
            )
        image = self.pixel_selection.image
        image[image > 0] = 255
        ski.io.imsave(filename, image)

    def clear_pixel_roi(self, *args) -> None:
        self.pixel_selection.setImage(
            np.zeros_like(
                self.pixel_selection.image
            )
        )

    def load_pixel_roi(self, *args, filename: str = None) -> None:
        if filename is None:
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                None, 'Image file', None, 'Tif files (*.tif)'
            )
        image = ski.io.imread(filename)
        self.pixel_selection.setImage(image)

    def image_changed(self, *args) -> None:
        # checkbox used to decide if the image should be plotted
        if self.checkBox_5.isChecked():
            image_name = self.comboBox_7.currentText()
            if image_name in self.clsm_representations.keys():
                image = self.clsm_representations.get(image_name, None)
                if image is not None:
                    # update spinBox that is used to select the
                    # current frame
                    self.set_current_frame_range(0, image.shape[0] - 1)

                    # select a gradient preset
                    self.hist.gradient.loadPreset(self.gradient_key)
                    if isinstance(image, np.ndarray):
                        # plot the sum of all frames
                        if self.plot_sum_of_frames:
                            self.current_image = image.sum(axis=0)
                            self.current_image_subset_1 = image[::2].sum(axis=0)
                            self.current_image_subset_2 = image[1::2].sum(axis=0)
                        # plot the mean of all frames
                        elif self.plot_mean_of_frames:
                            self.current_image = image.mean(axis=0)
                            self.current_image_subset_1 = image[::2].mean(axis=0)
                            self.current_image_subset_2 = image[1::2].mean(axis=0)
                        # plot only the currently selected frame
                        else:
                            frame_idx = self.current_frame
                            ref_idx = max(0, frame_idx - 1)
                            if (ref_idx == 0) and (frame_idx == 0):
                                ref_idx += 1
                            self.current_image = image[frame_idx]
                            self.current_image_subset_1 = image[frame_idx]
                            self.current_image_subset_2 = image[ref_idx]
                        data = self.current_image
                        self.update_frc_plot()
                        # pyqtgraph is column major by default
                        self.img.setImage(data)
                        # transpose image (row, column) -> (column, row)
                        self.pixel_selection.setImage(np.zeros_like(data))
                        self.hist.setLevels(data.min() + 1e-9, data.max())
                        self.brush_kernel *= max(data.flatten()) / max(1, max(self.brush_kernel.flatten()))
                        # zoom to fit image
                        self.img_plot.autoRange()

                        # Get template for roi
                        if not isinstance(self.pixel_selection.image, np.ndarray):
                            self.pixel_selection.setImage(np.zeros_like(data))
                        elif self.pixel_selection.image.shape != self.img.image.shape:
                            self.pixel_selection.setImage(np.zeros_like(data))
        # update roi
        if not self.checkBox_6.isChecked():
            w = self.pixel_selection.getPixmap()
            w.hide()
        self.update_plot()

    def save_image(self, *args, image_name: str = None, filename: str = None) -> None:
        if not isinstance(image_name, str):
            image_name = self.comboBox_7.currentText()
        if not isinstance(filename, str):
            filename = QtWidgets.QFileDialog.getSaveFileName(
                None, "Image file", None, 'Png files (*.png); Tif files (*.tif)'
            )

        if image_name in self.clsm_representations.keys():
            image = self.clsm_representations.get(image_name, None)
            if image is not None:
                if self.radioButton_4.isChecked():
                    data = image.sum(axis=0)
                else:
                    frame_idx = self.spinBox_8.value()
                    data = image[frame_idx]
                ski.io.imsave(filename, data)

    def update_plot(self, *args) -> None:
        if not isinstance(self.pixel_selection.image, np.ndarray):
            return
        sel = np.copy(self.pixel_selection.image)
        sel[sel > 0] = 1
        sel[sel < 0] = 0
        sel = sel.astype(dtype=np.uint8)
        clsm_image_object = self.clsm_images.get(self.current_clsm_name, None)
        if clsm_image_object:
            selection = np.ascontiguousarray(
                np.broadcast_to(
                    sel,
                    (
                        clsm_image_object.n_frames,
                        clsm_image_object.n_lines,
                        clsm_image_object.n_pixel
                    )
                )
            )
            decay = clsm_image_object.get_decay_of_pixels(
                tttr_data=self.tttr_data,
                mask=selection,
                tac_coarsening=self.tac_coarsening,
                stack_frames=self.stack_frames
            )
            header = self.tttr_data.get_header()
            x = np.arange(decay.shape[1])
            t = x * header.micro_time_resolution * self.tac_coarsening * 1e9 # units should be ns
            if self.stack_frames:
                y = decay.sum(axis=0)
            else:
                y = decay[self.current_frame]
            y_pos = np.where(y > 0)[0]
            if len(y_pos) > 0:
                i_y_max = y_pos[-1]
                y = y[:i_y_max]
                t = t[:i_y_max]
            ey = counting_noise(y)
            if chisurf:
                experiment = chisurf.experiment['TCSPC']
            else:
                experiment = None
            self.current_decay = DataCurve(x=t, y=y, ey=ey, experiment=experiment)
            self.plot_all_curves()

    def setup_image_plot(self, decay_plot: pyqtgraph.PlotWindow) -> pyqtgraph.GraphicsLayoutWidget:
        win = pyqtgraph.GraphicsLayoutWidget()
        # A plot area (ViewBox + axes) for displaying the image
        self.img_plot = win.addPlot(title="")
        self.img_plot.setAspectLocked(True)

        p1 = self.img_plot
        # Item for displaying image data
        p1.addItem(self.img)
        p1.addItem(self.pixel_selection)
        self.pixel_selection.setCompositionMode(
            QtGui.QPainter.CompositionMode_Plus
        )

        # Contrast/color control
        hist = self.hist
        hist.setImageItem(self.img)
        hist.gradient.loadPreset(
            next(iter(pyqtgraph.graphicsItems.GradientEditorItem.Gradients))
        )
        win.addItem(hist)

        win.setSizePolicy(
            QtWidgets.QSizePolicy(
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Expanding
            )
        )
        win.show()

        # Monkey-patch the image to use our custom hover function.
        # This is generally discouraged (you should subclass ImageItem instead),
        # but it works for a very simple use like this.
        def imagehoverEvent(event) -> None:
            """Show the position, pixel, and value under the mouse cursor.
            """
            if event.isExit():
                p1.setTitle("")
                return
            pos = event.pos()
            i, j = pos.y(), pos.x()
            i = int(np.clip(i, 0, self.img.image.shape[0] - 1))
            j = int(np.clip(j, 0, self.img.image.shape[1] - 1))
            val = self.img.image[i, j]
            ppos = self.img.mapToParent(pos)
            x, y = ppos.x(), ppos.y()
            p1.setTitle("pos: (%0.1f, %0.1f)  pixel: (%d, %d)  value: %g" % (x, y, i, j, val))

        self.pixel_selection.hoverEvent = imagehoverEvent

        # Monkey-patch the image to use our custom hover function.
        # This is generally discouraged (you should subclass ImageItem instead),
        # but it works for a very simple use like this.
        def imageMouseDragEvent(event) -> None:
            """Show the position, pixel, and value under the mouse cursor.
            """
            if event.button() != QtCore.Qt.LeftButton:
                return
            event.accept()
            self.pixel_selection.drawAt(event.pos(), event)
            if self.do_live_updates:
                self.update_plot()
        self.pixel_selection.mouseDragEvent = imageMouseDragEvent
        return win

    def update_brush(self, *args) -> None:
        self.brush_size = int(self.spinBox_2.value())
        self.brush_width = float(self.doubleSpinBox.value())
        self.brush_kernel = gaussian_kernel(self.brush_size, self.brush_width)
        self.pixel_selection.setDrawKernel(
            self.brush_kernel,
            mask=self.brush_kernel,
            center=(1, 1), mode='add'
        )
        if isinstance(self.img.image, np.ndarray):
            # data = self.img.image
            # self.brush_kernel *= 255.0 / max(self.brush_kernel.flatten())
            self.brush_kernel[self.brush_kernel> 0.001] = 30000
        # The brush is set to selection mode
        select = self.radioButton.isChecked()
        if not select:
            self.brush_kernel *= -1

    def update_clsm_settings(self, current_setup, clsm_settings):
        # frame marker
        frame_marker_text: str = ','.join([str(x) for x in clsm_settings[current_setup]['frame_marker']])
        self.lineEdit_2.setText(frame_marker_text)
        # line start
        self.spinBox_4.setValue(
            clsm_settings[current_setup]['line_start_marker']
        )
        # line stop
        self.spinBox_5.setValue(
            clsm_settings[current_setup]['line_stop_marker']
        )
        # event marker
        self.spinBox_6.setValue(
            clsm_settings[current_setup]['event_type_marker']
        )
        # number of pixel
        self.spinBox_7.setValue(
            clsm_settings[current_setup].get('pixel_per_line', 0)
        )
    def setup_changed(self, *args) -> None:
        current_setup = self.comboBox_5.currentText()
        tttr_type = clsm_settings[current_setup]['tttr_type']
        tttr_type_idx = self.comboBox_4.findText(tttr_type)
        self.comboBox_4.setCurrentIndex(tttr_type_idx)
        clsm_routine = clsm_settings[current_setup]['routine']
        clsm_routine_idx = self.comboBox.findText(clsm_routine)
        self.comboBox.setCurrentIndex(clsm_routine_idx)
        self.update_clsm_settings(current_setup, clsm_settings)

    def clear_images(self, *args) -> None:
        self.clsm_representations.clear()
        self.comboBox_7.clear()
        self.image_changed()

    def remove_image(self, *args, image_name: str = None) -> None:
        # Read from UI
        if image_name is None:
            image_name = self.comboBox_7.currentText()
        self.clsm_representations.pop(image_name, None)
        # Update UI
        self.comboBox_7.currentText()
        self.comboBox_7.removeItem(
            self.comboBox_7.currentIndex()
        )

    def add_clsm(
            self,
            *args,
            frame_marker: typing.List[int] = None,
            line_start_marker: int = None,
            line_stop_marker: int = None,
            event_type_marker: int = None,
            pixel_per_line: int = None,
            reading_routine: str = None,
            clsm_name: str = None,
            channels: typing.List[int] = None
    ) -> None:
        # Read from UI
        if not isinstance(frame_marker, list):
            frame_marker = [
                int(i) for i in str(self.lineEdit_2.text()).split(",")
            ]
            if len(frame_marker) == 0:
                frame_marker = None
        if line_start_marker is None:
            line_start_marker = int(self.spinBox_4.value())
        if line_stop_marker is None:
            line_stop_marker = int(self.spinBox_5.value())
        if event_type_marker is None:
            event_type_marker = int(self.spinBox_6.value())
        if pixel_per_line is None:
            pixel_per_line = int(self.spinBox_7.value())
        if reading_routine is None:
            reading_routine = str(self.comboBox.currentText())
        if clsm_name is None:
            fn, _ = os.path.splitext(self.filename)
            clsm_name = fn + "_ch(" + self.lineEdit_3.text() + ")"
        if channels is None:
            channels = [int(i) for i in str(self.lineEdit_3.text()).split(",")]

        clsm_image = tttrlib.CLSMImage(
            self.tttr_data,
            frame_marker,
            line_start_marker,
            line_stop_marker,
            event_type_marker,
            pixel_per_line,
            reading_routine
        )
        clsm_image.fill(
            tttr_data=self.tttr_data,
            channels=channels
        )

        # Update UI
        self.clsm_images[clsm_name] = clsm_image
        self.comboBox_9.clear()
        self.comboBox_9.addItems(
            list(self.clsm_images.keys())
        )

    def update_frc_plot(self):
        density, bins = compute_frc(
            self.current_image_subset_1,
            self.current_image_subset_2
        )
        self.frc_plot_window.clear()
        self.frc_plot_window.plot(x=bins, y=density)

    def remove_clsm(self, *args, clsm_name: str = None) -> None:
        # Read from UI
        if clsm_name is None:
            clsm_name = self.comboBox_9.currentText()

        self.clsm_images.pop(clsm_name, None)

        # Update UI
        self.comboBox_9.clear()
        self.comboBox_9.addItems(
            list(self.clsm_images.keys())
        )

    def __init__(self, *args, **kwargs) -> None:
        # initilize UI
        super().__init__(*args, **kwargs)
        uic.loadUi(
            os.path.join(
                os.path.dirname(
                    os.path.abspath(__file__)
                ),
                "gui.ui"
            ),
            self
        )
        self.groupBox_2.setChecked(False)

        ##########################################################
        #      Arrange Docks and window positions                #
        #      Window-controls tile, stack etc.                  #
        ##########################################################
        # self.tabifyDockWidget(self.dockWidget, self.dockWidget_4)
        # self.tabifyDockWidget(self.dockWidget_3, self.dockWidget_4)

        # FRC plot
        self.frc_plot_window = pyqtgraph.PlotWidget()
        self.verticalLayout_15.addWidget(self.frc_plot_window)

        # Used reading routines
        self.current_setup = list(clsm_settings.keys())[0]
        self.comboBox_5.addItems(
            list(clsm_settings.keys())
        )
        routines = list()
        for k in clsm_settings:
            routines.append(clsm_settings[k]['routine'])
        self.comboBox.addItems(
            list(set(routines))
        )

        # image and pixel selection
        self.img = pyqtgraph.ImageItem()
        self.pixel_selection = pyqtgraph.ImageItem()
        self.hist = pyqtgraph.HistogramLUTItem()
        self.gradient_key = 9, list(pyqtgraph.graphicsItems.GradientEditorItem.Gradients.keys())
        self._curves = imported_datasets

        self.update_brush()

        # Add curve experiment curve selector
        if chisurf is None:
            self.cs = ExperimentalDataSelector(
                get_data_sets=self.get_data_curves,
                click_close=False
            )
        else:
            if chisurf:
                experiment = chisurf.experiment['TCSPC']
            else:
                experiment = None
            self.cs = chisurf.gui.widgets.experiments.ExperimentalDataSelector(
                parent=None,
                change_event=self.update_plot,
                experiment=experiment
            )
        self.verticalLayout_9.addWidget(self.cs)

        # Add plot of image
        self.plot = pyqtgraph.PlotWidget()
        plot = self.plot.getPlotItem()
        self.verticalLayout.addWidget(self.plot)
        self.legend = plot.addLegend()
        plot_item = self.plot.getPlotItem()
        plot_item.setLogMode(x=False, y=False)
        self.cs.onRemoveDataset = self.onRemoveDataset

        # Add image view
        image_widget = self.setup_image_plot(
            decay_plot=self.plot
        )
        self.verticalLayout_4.addWidget(image_widget)

        # Signal slots
        self.actionLoad_file.triggered.connect(self.open_tttr_file)
        self.actionchange_brush_size.triggered.connect(self.update_brush)
        self.actionSave_pixel_mask.triggered.connect(self.save_pixel_roi)
        self.actionClear_pixel_mask.triggered.connect(self.clear_pixel_roi)
        self.actionLoad_pixel_mask.triggered.connect(self.load_pixel_roi)
        self.actionAdd_decay.triggered.connect(self.add_curve)
        self.actionSetup_changed.triggered.connect(self.setup_changed)
        self.actionAdd_image.triggered.connect(self.add_representation)
        self.actionImage_changed.triggered.connect(self.image_changed)
        self.actionUpdate_plot.triggered.connect(self.update_plot)
        self.actionSave_image.triggered.connect(self.save_image)
        self.actionClear_images.triggered.connect(self.clear_images)
        self.actionAdd_mask.triggered.connect(self.add_roi)
        self.actionMask_changed.triggered.connect(self.roi_changed)
        self.actionRemove_image.triggered.connect(self.remove_image)
        self.actionRemove_current_mask.triggered.connect(self.remove_rois)
        self.actionAdd_CLSM.triggered.connect(self.add_clsm)
        self.actionRemove_CLSM.triggered.connect(self.remove_clsm)

        # update the UI
        self.setup_changed()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = CLSMPixelSelect()
    win.show()
    sys.exit(app.exec_())




