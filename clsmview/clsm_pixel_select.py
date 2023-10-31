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
try:
    import pathlib
except ImportError:
    import pathlib2 as pathlib

import pyqtgraph
import pyqtgraph.dockarea
import pyqtgraph.graphicsItems.GradientEditorItem
from qtpy import QtCore, QtGui, QtWidgets, uic

import numpy as np
import pandas as pd
import cv2

import tttrlib
import scikit_fluorescence as skf
import scikit_fluorescence.math.signal
import scikit_fluorescence.decay.tcspc

try:
    import chisurf
    import chisurf.settings
    from chisurf.gui.widgets.experiments import ExperimentalDataSelector
    plot_settings = chisurf.settings.gui['plot']
    imported_datasets = chisurf.imported_datasets
except ImportError:
    from . selector import ExperimentalDataSelector
    with open(pathlib.Path(__file__).parent / 'settings.yaml') as fp:
        settings = yaml.safe_load(fp)
        plot_settings = settings['gui']['plot']
    imported_datasets = list()

clsm_settings = yaml.safe_load(
    open(
        pathlib.Path(__file__).parent / pathlib.Path("clsm_settings.yaml")
    )
)


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
        The kernel that is used to brush an selection. The brush kernel is a
        Gaussian with a width defined by brush_width
    brush_size : int
        The size of the brush kernel
    brush_width : float
        Defines the v
    masks : dict
        The user can define a set of masks. The masks dictionary stores these
        user defined masks

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
    masks: typing.Dict[str, np.ndarray] = dict()

    img_plot: pyqtgraph.PlotWidget = None
    frc_plot_window: pyqtgraph.PlotWidget = None
    current_decay = pd.DataFrame()  # type: pd.DataFrame

    def get_data_curves(self, *args, **kwargs):
        #  type: (typing.List, typing.Dict) -> typing.List[pd.DataFrame]
        return self._curves

    @property
    def curve_name(self):
        s = str(self.lineEdit.text())
        if len(s) == 0:
            return "no-name"
        else:
            return s

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

    def clear_curves(self, *args):
        # type: (typing.List) -> ()
        self._curves.clear()
        plot = self.plot.getPlotItem()
        plot.clear()
        self.cs.update()

    def remove_curve(self, selected_index: typing.List[int] = None):
        # Read from UI
        if selected_index is None:
            selected_index = [
                i.row() for i in self.cs.selectedIndexes()
            ]
        curves = list()
        for idx, curve in enumerate(curves):
            if idx in selected_index:
                curves.append(curve)
        self._curves = curves
        # Update UI
        self.cs.update()
        self.plot_all_curves()

    def update_selection_plot(self):
        plot = self.plot.getPlotItem()
        curve = plot.curves[0]
        curve.setData(
            x=self.current_decay['x'],
            y=self.current_decay['y']
        )

    def plot_all_curves(
            self
    ) -> None:
        """Plots and updates the curves
        """
        plot = self.plot.getPlotItem()
        plot.clear()
        curve = self.current_decay
        plot.plot(
            x=curve['x'], y=curve['y'],
            name="Current selection"
        )

        self.legend = plot.addLegend()
        # plot.setLogMode(x=False, y=True)
        plot.showGrid(True, True, 1.0)

        current_curve = self.cs.selected_curve_index
        lw = plot_settings['line_width']
        for i, curve in enumerate(self._curves):
            w = lw * 0.5 if i != current_curve else 2.5 * lw
            plot.plot(
                x=curve['x'], y=curve['y'],
                pen=pyqtgraph.mkPen(
                    # chisurf.settings.colors[i % len(chisurf.settings.colors)]['hex'],
                    width=w
                ),
                name=curve.name
            )
        # plot.autoRange()

    def add_curve(self, *args, v=None, **kwargs):
        # type: (typing.List, pd.DataFrame, typing.Dict) -> None
        decay = v if isinstance(v, pd.DataFrame) else copy.copy(self.current_decay)
        try:
            name = self.listWidget.currentItem().text()
        except AttributeError:
            name = self.lineEdit_5.text()
        if len(name) == 0:
            name = 'no-name'
        decay.name = name
        self._curves.append(decay)

        # Update UI
        self.cs.update()
        self.plot_all_curves()

    def open_tttr_file(
            self,
            *args,
            filename: str = None,
            tttr_type: str = None
    ) -> None:
        """Opens a tttr file and sets the attribute '.tttr_data'.
        If no filename a file selection window is used to find a tttr file.
        If no tttr_type is specified the values provided by the UI are used.

        :param filename: (optional) parameter specifying the filename
        :param tttr_type: (optional) parameter specifying the tttr type
        :return:
        """
        # Load from UI
        if not isinstance(filename, str):
            tentative_filename = str(self.lineEdit.text())
            if os.path.isfile(tentative_filename):
                filename = tentative_filename
            else:
                filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                    None,
                    "TTTR CLSM file",
                    None,
                    "*.ht3;*.ptu;*.spc"
                )

        if tttr_type is None:
            tttr_type = str(self.comboBox_4.currentText())
        if os.path.isfile(filename):
            self.lineEdit.setText(filename)
            # Load TTTR data
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
        """Create a new representation of an CLSM image. If no clsm_image
        of the type tttrlib.CLSMImage is provided the clsm_image is taken from
        self.clsm_images using the parameter clsm_name.

        :param clsm_image:
        :param image_type:
        :param clsm_name: the dictionary key of the CLSM image from which
        a new representation is generated for.
        :param n_ph_min: the minimum number of photons in a pixel. This
        value is used for mean micro time representations to discriminate
        pixels with few photons.

        :return:
        """
        # Read from UI
        if not isinstance(clsm_name, str):
            clsm_name = str(self.comboBox_9.currentText())
        if not isinstance(clsm_name, np.ndarray):
            clsm_image = self.clsm_images[clsm_name]
        if not isinstance(image_type, str):
            image_type = str(self.comboBox_3.currentText())
        if not isinstance(n_ph_min, int):
            n_ph_min = int(self.spinBox.value())
        if not isinstance(image_name, str):
            image_name = str(self.lineEdit_4.text())

        if image_type == "Mean micro time":
            n_ph_min = int(self.spinBox.value())
            mean_micro_time = clsm_image.get_mean_micro_time_image(self.tttr_data, n_ph_min, False)
            data = mean_micro_time.astype(np.float64)
        elif image_type == "Intensity":
            data = clsm_image.intensity.astype(np.float64)
        else:
            mean_micro_time = clsm_image.get_mean_micro_time_image(
                self.tttr_data,
                n_ph_min,
                False
            )
            intensity_image = clsm_image.get_intensity_image()
            data = mean_micro_time * intensity_image
        self.clsm_representations[image_name] = data

        # update UI
        self.comboBox_7.clear()
        self.comboBox_7.addItems(
            list(self.clsm_representations.keys())
        )
        self.image_changed()

    def add_mask(self, *args, mask_name: str = None) -> None:
        if mask_name is None:
            mask_name = self.lineEdit_5.text()
        self.masks[mask_name] = self.pixel_selection.image

        # Update UI
        self.listWidget.clear()
        for mn in self.masks.keys():
            self.listWidget.addItem(mn)

    def mask_changed(self, *args) -> None:
        mask_name = self.listWidget.currentItem().text()
        d = self.masks[mask_name]
        d *= np.max(self.img.image.flatten())
        self.pixel_selection.setImage(d)
        self.pixel_selection.updateImage()
        self.update_plot()

    def get_selected_masks(
            self
    ) -> typing.List[str]:
        """Returns a list of mask names that are currently
        selected in the UI.

        :return: A list of mask names
        """
        # Get list of mask_names from UI
        mask_names = list()
        list_items = self.listWidget.selectedItems()
        if not list_items:
            return list()
        for item in list_items:
            lw = self.listWidget.takeItem(self.listWidget.row(item))
            mask_names.append(lw.text())
        return mask_names

    def remove_masks(
            self,
            *args,
            mask_names: typing.List[str] = None
    ) -> None:
        # Read from UI
        if mask_names is None:
            mask_names = self.get_selected_masks()
        # Remove masks
        for mask_name in mask_names:
            self.masks.pop(mask_name, None)

    def save_pixel_mask(
            self,
            event,
            filename: str = None
    ) -> None:
        if filename is None:
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                None, 'Image file', None, 'All files (*.png)'
            )
        print("save_pixel_mask")
        print(filename)
        image = self.pixel_selection.image
        image[image > 0] = 255
        cv2.imwrite(filename, image)

    def clear_pixel_mask(
            self,
            *args
    ) -> None:
        self.pixel_selection.setImage(
            np.zeros_like(
                self.pixel_selection.image
            )
        )

    def load_pixel_mask(
            self,
            *args,
            filename: str = None
    ) -> None:
        if filename is None:
            filename = QtWidgets.QFileDialog.getOpenFileName(
                None, 'Image file', None, 'All files (*.png)'
            )
        image = cv2.imread(filename)
        self.pixel_selection.setImage(
            image=cv2.cvtColor(
                image, cv2.COLOR_BGR2GRAY
            ).astype(np.float64)
        )

    def image_changed(
            self,
            *args
    ) -> None:
        # checkbox used to decide if the image should be plotted
        if self.checkBox_5.isChecked():
            image_name = self.comboBox_7.currentText()
            if image_name in self.clsm_representations.keys():
                image = self.clsm_representations.get(image_name, None)
                if image is not None:
                    # update spinBox that is used to select the
                    # current frame
                    self.spinBox_8.setMinimum(0)
                    self.spinBox_8.setMaximum(image.shape[0] - 1)

                    # select a gradient preset
                    self.hist.gradient.loadPreset(
                        self.comboBox_2.currentText()
                    )
                    if isinstance(image, np.ndarray):
                        # plot the sum of all frames
                        if self.radioButton_4.isChecked():
                            self.current_image = image.sum(axis=0)
                            self.current_image_subset_1 = image[::2].sum(axis=0)
                            self.current_image_subset_2 = image[1::2].sum(axis=0)
                        # plot the mean of all frames
                        elif self.radioButton_5.isChecked():
                            self.current_image = image.mean(axis=0)
                            self.current_image_subset_1 = image[::2].mean(axis=0)
                            self.current_image_subset_2 = image[1::2].mean(axis=0)
                        # plot only the currently selected frame
                        else:
                            frame_idx = self.spinBox_8.value()
                            ref_idx = max(0, frame_idx - 1)
                            if (ref_idx == 0) and (frame_idx == 0):
                                ref_idx += 1
                            self.current_image = image[frame_idx]
                            self.current_image_subset_1 = image[frame_idx]
                            self.current_image_subset_2 = image[ref_idx]
                        data = self.current_image
                        self.update_frc_plot()
                        # pyqtgraph is column major by default
                        self.img.setImage(data.T)
                        # transpose image (row, column) -> (column, row)
                        self.pixel_selection.setImage(np.zeros_like(data))
                        self.hist.setLevels(data.min() + 1e-9, data.max())
                        self.brush_kernel *= max(data.flatten()) / max(1, max(self.brush_kernel.flatten()))
                        # zoom to fit image
                        self.img_plot.autoRange()

                        # Get template for mask
                        if not isinstance(self.pixel_selection.image, np.ndarray):
                            self.pixel_selection.setImage(np.zeros_like(data))
                        elif self.pixel_selection.image.shape != self.img.image.shape:
                            self.pixel_selection.setImage(np.zeros_like(data))
        else:
            self.img.setImage(
                np.zeros((512, 512))
            )
        # update mask
        if not self.checkBox_6.isChecked():
            w = self.pixel_selection.getPixmap()
            w.hide()
        self.update_plot()

    def save_image(
            self,
            *args,
            image_name: str = None,
            filename: str = None
    ) -> None:
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
                cv2.imwrite(filename, data.T)

    def update_plot(
            self,
            *args
    ) -> None:
        if not isinstance(self.pixel_selection.image, np.ndarray):
            return
        # Transpose sel (pyqtgraph is column major)
        # (column, row) -> (row , column)
        sel = np.copy(self.pixel_selection.image).T
        sel[sel > 0] = 1
        sel[sel < 0] = 0
        sel = sel.astype(dtype=np.uint8)
        clsm_name = self.comboBox_9.currentText()
        clsm_image_object = self.clsm_images.get(clsm_name, None)
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
            tac_coarsening = int(self.comboBox_6.currentText())
            stack_frames = self.radioButton_4.isChecked() or self.radioButton_5.isChecked()
            decay = clsm_image_object.get_decay_of_pixels(
                tttr_data=self.tttr_data,
                mask=selection,
                tac_coarsening=tac_coarsening,
                stack_frames=stack_frames
            )
            header = self.tttr_data.get_header()
            x = np.arange(decay.shape[1])
            t = x * header.micro_time_resolution * tac_coarsening
            if stack_frames:
                y = decay.sum(axis=0)
            else:
                y = decay[self.spinBox_8.value()]
            y_pos = np.where(y > 0)[0]
            if len(y_pos) > 0:
                i_y_max = y_pos[-1]
                y = y[:i_y_max]
                t = t[:i_y_max]
            self.current_decay = pd.DataFrame()
            self.current_decay["x"] = t
            self.current_decay["y"] = y
            self.current_decay["ey"] = skf.decay.tcspc.counting_noise(y)
            self.plot_all_curves()

    def setup_image_plot(
            self,
            decay_plot: pyqtgraph.PlotWindow
    ) -> pyqtgraph.GraphicsLayoutWidget:
        win = pyqtgraph.GraphicsLayoutWidget()
        # A plot area (ViewBox + axes) for displaying the image
        self.img_plot = win.addPlot(title="")
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
        def imagehoverEvent(
                event
        ) -> None:
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
        def imageMouseDragEvent(
                event
        ) -> None:
            """Show the position, pixel, and value under the mouse cursor.
            """
            if event.button() != QtCore.Qt.LeftButton:
                return
            elif self.pixel_selection.drawKernel is not None:
                # draw on the image
                event.accept()
                self.pixel_selection.drawAt(event.pos(), event)
                if self.checkBox_4.isChecked():
                    self.update_plot()
        self.pixel_selection.mouseDragEvent = imageMouseDragEvent
        return win

    def update_brush(
            self,
            *args,
    ) -> None:
        self.brush_size = int(self.spinBox_2.value())
        self.brush_width = float(self.doubleSpinBox.value())
        self.brush_kernel = skf.math.signal.gaussian_kernel(
                    self.brush_size,
                    self.brush_width
                )

        self.pixel_selection.setDrawKernel(
            self.brush_kernel,
            mask=self.brush_kernel,
            center=(1, 1), mode='add'
        )
        if isinstance(self.img.image, np.ndarray):
            # data = self.img.image
            self.brush_kernel *= 255.0 / max(self.brush_kernel.flatten())

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
            clsm_name = str(self.lineEdit_6.text())
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
        density, bins = skf.math.signal.compute_frc(
            self.current_image_subset_1,
            self.current_image_subset_2
        )
        self.frc_plot_window.plot(
        )
        self.frc_plot_window.clear()
        self.frc_plot_window.plot(
            x=bins, y=density,
        )

    def remove_clsm(
            self,
            *args,
            clsm_name: str = None
    ) -> None:
        # Read from UI
        if clsm_name is None:
            clsm_name = self.comboBox_9.currentText()

        self.clsm_images.pop(clsm_name, None)

        # Update UI
        self.comboBox_9.clear()
        self.comboBox_9.addItems(
            list(self.clsm_images.keys())
        )

    def __init__(
            self,
            *args,
            **kwargs
    ) -> None:
        # initilize UI
        super().__init__(
            *args,
            **kwargs
        )
        uic.loadUi(
            os.path.join(
                os.path.dirname(
                    os.path.abspath(__file__)
                ),
                "clsm_pixel_select.ui"
            ),
            self
        )

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
        self.comboBox_2.addItems(
            list(pyqtgraph.graphicsItems.GradientEditorItem.Gradients.keys())
        )

        self._curves = imported_datasets
        self.update_brush()

        # Add curve experiment curve selector
        self.cs = ExperimentalDataSelector(
            get_data_sets=self.get_data_curves,
            click_close=False
        )
        self.verticalLayout_9.addWidget(self.cs)

        # Add plot of image
        self.plot = pyqtgraph.PlotWidget()
        plot = self.plot.getPlotItem()
        self.verticalLayout.addWidget(self.plot)
        self.legend = plot.addLegend()
        plot_item = self.plot.getPlotItem()
        plot_item.setLogMode(
            x=False,
            y=False
        )
        self.cs.onRemoveDataset = self.onRemoveDataset

        # Add image view
        image_widget = self.setup_image_plot(
            decay_plot=self.plot
        )
        self.verticalLayout_4.addWidget(image_widget)

        # Signal slots
        self.actionLoad_file.triggered.connect(self.open_tttr_file)
        self.actionchange_brush_size.triggered.connect(self.update_brush)
        self.actionSave_pixel_mask.triggered.connect(self.save_pixel_mask)
        self.actionClear_pixel_mask.triggered.connect(self.clear_pixel_mask)
        self.actionLoad_pixel_mask.triggered.connect(self.load_pixel_mask)
        self.actionAdd_decay.triggered.connect(self.add_curve)
        self.actionClear_decay_curves.triggered.connect(self.clear_curves)
        self.actionSetup_changed.triggered.connect(self.setup_changed)
        self.actionAdd_image.triggered.connect(self.add_representation)
        self.actionImage_changed.triggered.connect(self.image_changed)
        self.actionUpdate_plot.triggered.connect(self.update_plot)
        self.actionSave_image.triggered.connect(self.save_image)
        self.actionClear_images.triggered.connect(self.clear_images)
        self.actionAdd_mask.triggered.connect(self.add_mask)
        self.actionMask_changed.triggered.connect(self.mask_changed)
        self.actionRemove_image.triggered.connect(self.remove_image)
        self.actionRemove_current_mask.triggered.connect(self.remove_masks)
        self.actionAdd_CLSM.triggered.connect(self.add_clsm)
        self.actionRemove_CLSM.triggered.connect(self.remove_clsm)

        # update the UI
        self.setup_changed()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = CLSMPixelSelect()
    win.show()
    sys.exit(app.exec_())




