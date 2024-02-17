from __future__ import print_function

import typing
from typing import Callable

import os
import pandas as pd

from qtpy import QtWidgets, QtCore, QtGui


class ExperimentalDataSelector(QtWidgets.QTreeWidget):

    _data_sets: typing.List[pd.DataFrame] = list()

    @property
    def curve_name(self) -> str:
        try:
            return self.selected_dataset.name
        except AttributeError:
            return "Noname"

    @property
    def datasets(self) -> typing.List[pd.DataFrame]:
        data_curves = self.get_data_sets(curve_type=self.curve_type)
        return data_curves

    @property
    def selected_curve_index(self) -> int:
        if self.currentIndex().parent().isValid():
            return self.currentIndex().parent().row()
        else:
            return self.currentIndex().row()

    @selected_curve_index.setter
    def selected_curve_index(self, v: int) -> None:
        self.setCurrentItem(self.topLevelItem(v))

    @property
    def selected_dataset(self) -> pd.DataFrame:
        return self.datasets[self.selected_curve_index]

    @property
    def selected_datasets(self) -> typing.List[pd.DataFrame]:
        data_sets_idx = self.selected_dataset_idx
        return [self.datasets[i] for i in data_sets_idx]

    @property
    def selected_dataset_idx(self) -> typing.List[int]:
        return [r.row() for r in self.selectedIndexes()]

    def onCurveChanged(self):
        if self.click_close:
            self.hide()
        self.change_event()

    def onRemoveDataset(self):
        dataset_idx = [
            selected_index.row() for selected_index in self.selectedIndexes()
        ]
        data_sets = list()
        for i, d in enumerate(self.datasets):
            if i not in dataset_idx:
                data_sets.append(d)
        self._data_sets = data_sets
        self.update(update_others=True)

    def onSaveDataset(self, filename: str = None) -> None:
        if not isinstance(filename, str):
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(None, "", None, "*.*")
        base_name, extension = os.path.splitext(filename)
        if extension.lower() == '.csv':
            self.selected_dataset.to_csv(filename, sep='\t')
        else:
            filename = base_name + '.json'
            self.selected_dataset.to_json(filename)

    def contextMenuEvent(self, event):
        if self.context_menu_enabled:
            menu = QtWidgets.QMenu(self)
            menu.setTitle("Datasets")
            menu.addAction("Save").triggered.connect(self.onSaveDataset)
            menu.addAction("Remove").triggered.connect(self.onRemoveDataset)
            menu.addAction("Refresh").triggered.connect(self.update)
            menu.exec_(event.globalPos())

    def update(self, *args, **kwargs):
        super().update()
        try:
            window_title = self.fit.name
            self.setWindowTitle(window_title)
        except AttributeError:
            self.setWindowTitle("")
        self.clear()

        for d in self.datasets:
            fn = d.name
            widget_name = os.path.basename(fn)
            item = QtWidgets.QTreeWidgetItem(self, [widget_name])
            item.setToolTip(0, fn)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)

    def onItemChanged(self):
        if self.selected_datasets:
            ds = self.selected_datasets[0]
            ds.name = str(self.currentItem().text(0))
            self.update(update_others=True)

    def change_event(self):
        pass

    def show(self):
        self.update()
        QtWidgets.QTreeWidget.show(self)

    def __init__(
            self,
            drag_enabled: bool = False,
            click_close: bool = True,
            change_event: Callable = None,
            curve_types: str = 'experiment',
            get_data_sets: Callable = None,
            parent: QtWidgets.QWidget = None,
            icon: QtGui.QIcon = None,
            context_menu_enabled: bool = True
    ):
        if get_data_sets is None:
            def get_data_sets(**kwargs):
                return self._data_sets
            self.get_data_sets = get_data_sets
        else:
            self.get_data_sets = get_data_sets

        if change_event is not None:
            self.change_event = change_event

        if icon is None:
            icon = QtGui.QIcon(":/icons/icons/list-add.png")

        self.curve_type = curve_types
        self.click_close = click_close
        self.context_menu_enabled = context_menu_enabled

        super().__init__(parent)

        self.setWindowIcon(icon)
        self.setWordWrap(True)
        self.setHeaderHidden(True)
        self.setAlternatingRowColors(True)

        if drag_enabled:
            self.setAcceptDrops(True)
            self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)

        # http://python.6.x6.nabble.com/Drag-and-drop-editing-in-QListWidget-or-QListView-td1792540.html
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.drag_item = None
        self.drag_row = None

        self.clicked.connect(self.onCurveChanged)
        self.itemChanged.connect(self.onItemChanged)

