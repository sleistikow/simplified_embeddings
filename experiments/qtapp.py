import os
import sys
import math

import pandas as pd
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QSlider, QComboBox, QVBoxLayout, QHBoxLayout, QWidget, QLabel, \
    QSizePolicy, QCheckBox
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from zipfile import ZipFile
from PIL import Image
from PIL.ImageQt import ImageQt, toqpixmap

from experiments.experiments_vmv import exp_name_from_params, exp_test_all_parameters
from statistics import plot_stress_experiment


class MainWindow(QWidget):
    def __init__(self, result_path, parameter_ranges):
        QWidget.__init__(self)

        self.parameter_ranges = parameter_ranges

        self.archive_path = os.path.join(result_path, 'figures.zip')
        self.stats_path = os.path.join(result_path, 'stats.csv')

        self.df = pd.read_csv(self.stats_path)

        self.setWindowTitle('Experiments')
        self.setMinimumSize(QSize(800, 600))

        top_row = QHBoxLayout()
        self.result_view = self.create_result_view()
        top_row.addWidget(self.result_view)
        self.stress = None

        self.checkbox_fix_param_t = None
        self.checkbox_fix_param_s = None
        self.checkbox_fix_param_a = None
        self.checkbox_fix_param_k = None
        self.slider_num_timesteps = None
        self.slider_scalings = None
        self.slider_weights = None
        self.slider_kernel_width = None
        self.slider_alphas = None
        top_row.addWidget(self.create_control_panel())
        self.update_result_view()

        bottom_row = QHBoxLayout()
        bottom_row.addWidget(self.create_2D_view())
        self.our_view = self.create_our_view()
        self.update_our_view()
        bottom_row.addWidget(self.our_view)
#        bottom_row.addWidget(self.create_1D_time_view())

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_row)
        main_layout.addLayout(bottom_row)

        self.setLayout(main_layout)

    def load_image(self, archive_path, image_name):
        with ZipFile(archive_path) as archive:
            with archive.open(image_name) as file:
                return toqpixmap(Image.open(file))

    def create_result_view(self):
        widget = QWidget()
        widget.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        return widget

    @staticmethod
    def set_widget(parent, widget):
        if parent.layout():
            for i in reversed(range(parent.layout().count())):
                parent.layout().itemAt(i).widget().setParent(None)

            parent.layout().addWidget(widget)

        else:
            layout = QVBoxLayout()
            layout.addWidget(widget)
            parent.setLayout(layout)

    def update_result_view(self, _=None):

        #stress_types = [self.stress.currentText()]
        stress_types = ['full', 'target']
        # scaling = self.exp_scalings[self.slider_scalings.sliderPosition()]

        param_t = None
        if self.checkbox_fix_param_t.isChecked():
            param_t = self.slider_num_timesteps.sliderPosition()

        param_s = None
        if self.checkbox_fix_param_s.isChecked():
            param_s = self.slider_scalings.sliderPosition()

        selected_weight = self.slider_weights.sliderPosition()
        #selected_weight = None
        figure = plot_stress_experiment(self.df, stress_types, selected_weight, param_t=param_t, param_s=param_s)

        canvas = FigureCanvas(figure)
        self.set_widget(self.result_view, canvas)

    def create_control_panel(self):

        layout = QVBoxLayout()

        label = QLabel()
        label.setText('Experiment:')
        layout.addWidget(label)

        self.checkbox_fix_param_t = QCheckBox('Fix param_t')
        self.checkbox_fix_param_t.stateChanged.connect(self.update_result_view)
        self.checkbox_fix_param_s = QCheckBox('Fix param_s')
        self.checkbox_fix_param_s.stateChanged.connect(self.update_result_view)
        self.checkbox_fix_param_a = QCheckBox('Fix param_a')
        self.checkbox_fix_param_a.stateChanged.connect(self.update_result_view)
        self.checkbox_fix_param_k = QCheckBox('Fix param_k')
        self.checkbox_fix_param_k.stateChanged.connect(self.update_result_view)

        exp_layout = QHBoxLayout()
        exp_layout.addWidget(self.checkbox_fix_param_t)
        exp_layout.addWidget(self.checkbox_fix_param_s)
        exp_layout.addWidget(self.checkbox_fix_param_a)
        exp_layout.addWidget(self.checkbox_fix_param_k)
        layout.addLayout(exp_layout)

        self.stress = QComboBox()
        self.stress.addItem('full')
        self.stress.addItem('time')
        self.stress.addItem('ref')
        self.stress.addItem('ref+time')
        self.stress.addItem('target')
        self.stress.currentTextChanged.connect(self.update_result_view)
        layout.addWidget(self.stress)

        label = QLabel()
        label.setText('Weight:')
        layout.addWidget(label)

        self.slider_weights = QSlider()
        self.slider_weights.setOrientation(1)
        self.slider_weights.setRange(0, len(self.parameter_ranges['param_w']) - 1)
        self.slider_weights.setValue(0)
        #self.weights.setTracking(False)
        self.slider_weights.valueChanged.connect(self.update_result_view)
        self.slider_weights.valueChanged.connect(self.update_our_view)
        layout.addWidget(self.slider_weights)

        label = QLabel()
        label.setText('Scaling:')
        layout.addWidget(label)

        self.slider_scalings = QSlider()
        self.slider_scalings.setOrientation(1)
        self.slider_scalings.setRange(0, len(self.parameter_ranges['param_s']) - 1)
        self.slider_scalings.setValue(0)
        # self.slider_scalings.setTracking(False)
        self.slider_scalings.valueChanged.connect(self.update_result_view)
        self.slider_scalings.valueChanged.connect(self.update_our_view)
        layout.addWidget(self.slider_scalings)

        label = QLabel()
        label.setText('Num Time Steps:')
        layout.addWidget(label)

        self.slider_num_timesteps = QSlider()
        self.slider_num_timesteps.setOrientation(1)
        self.slider_num_timesteps.setRange(0, len(self.parameter_ranges['param_t']) - 1)
        self.slider_num_timesteps.setValue(0)
        self.slider_num_timesteps.valueChanged.connect(self.update_result_view)
        self.slider_num_timesteps.valueChanged.connect(self.update_our_view)
        layout.addWidget(self.slider_num_timesteps)

        label = QLabel()
        label.setText('Kernel Width:')
        layout.addWidget(label)

        self.slider_kernel_width = QSlider()
        self.slider_kernel_width.setOrientation(1)
        self.slider_kernel_width.setRange(0, len(self.parameter_ranges['param_k']) - 1)
        self.slider_kernel_width.setValue(0)
        self.slider_kernel_width.valueChanged.connect(self.update_our_view)
        layout.addWidget(self.slider_kernel_width)

        label = QLabel()
        label.setText('Alpha:')
        layout.addWidget(label)

        self.slider_alphas = QSlider()
        self.slider_alphas.setOrientation(1)
        self.slider_alphas.setRange(0, len(self.parameter_ranges['param_a']) - 1)
        self.slider_alphas.setValue(0)
        self.slider_alphas.valueChanged.connect(self.update_our_view)
        layout.addWidget(self.slider_alphas)

        widget = QWidget()
        widget.setLayout(layout)

        return widget

    def create_2D_view(self):
        widget = QLabel()
        pixmap = self.load_image(self.archive_path, 'Exp Classical MDS - 2D Embedding.png')
        pixmap = pixmap.scaled(widget.size(), aspectRatioMode=True)
        widget.setPixmap(pixmap)
        return widget

    def create_our_view(self):
        widget = QLabel()
        return widget

    def update_our_view(self, _=None):

        params = {}
        params['param_t'] = self.parameter_ranges['param_t'][self.slider_num_timesteps.sliderPosition()]
        params['param_s'] = self.parameter_ranges['param_s'][self.slider_scalings.sliderPosition()]
        params['param_a'] = self.parameter_ranges['param_a'][self.slider_alphas.sliderPosition()]
        params['param_w'] = self.parameter_ranges['param_w'][self.slider_weights.sliderPosition()]
        params['param_k'] = self.parameter_ranges['param_k'][self.slider_kernel_width.sliderPosition()]

        exp_name = exp_name_from_params(params, self.parameter_ranges)

        experiment_str = f'Exp {exp_name} - 2D Embedding.png'

        pixmap = self.load_image(self.archive_path, experiment_str)
        pixmap = pixmap.scaled(self.our_view.size(), aspectRatioMode=True)
        self.our_view.setPixmap(pixmap)

    def create_1D_time_view(self):
        widget = QLabel()
        pixmap = self.load_image(self.archive_path, 'Exp Classical MDS - 1st PC + Time (corrected).png')
        widget.setPixmap(pixmap)
        return widget


if __name__ == '__main__':

    #path = '/data/ownCloud/projects/linearizedembedding/results/results_toy_2d'
    #path = '/home/spider/sciebo/projects/linearizedembedding/results/results_paper/'
    #path = '/data/ownCloud/projects/linearizedembedding/results/results_paper/'
    path = 'results/'

    app = QApplication(sys.argv)
    window = MainWindow(path, exp_test_all_parameters())
    window.show()
    app.exec()