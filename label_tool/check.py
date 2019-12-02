#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 13:55:41 2019

@author: anup
"""

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QFileDialog
import pandas as pd
import os
import sys
import pyfilename
import glob
import json
import time
import shutil
import random
import pickle

elevation_items = ['Ceiling',
                   'Gutter',
                   'Roof',
                   'Stucco',
                   'Vent',
                   'Downspout',
                   'Door',
                   'Windows']

elevation_rows = ['_'.join(i.lower().split()) for i in elevation_items]

damage_status_pre = ['No Damage','Peril Damage','Non Peril Damage']
damage_status = ['_'.join(i.lower().split()) for i in damage_status_pre]

dataset_rows = []
dataset_rows_dict = {}
for h, i in enumerate(elevation_rows):
    for k, j in enumerate(damage_status):
        dataset_rows.append(i + '_'+ j)
        dataset_rows_dict[i + '_'+ j] = [elevation_items[h], damage_status_pre[k]]

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

class ExampleApp(QtWidgets.QMainWindow, pyfilename.Ui_SmartLabeller):
    def __init__(self, parent=None):
        super(ExampleApp, self).__init__(parent)
        self.setupUi(self)
        self.current_row = 0
        

    def onboard_json(self):
        self.json_file_name = self.lineEdit.text()
        if self.json_file_name:
            if self.json_file_name[-5:] != '.json':
                self.json_file_name = self.json_file_name + '.json'
            with open(self.json_file_name) as json_file:
                self.image_profile = json.load(json_file)


    def mybutton_clicked(self):
        self.onboard_json()
        k = QtWidgets.QFileDialog
# File Selection
#        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*)", options=options)
        folder_path = str(k.getExistingDirectory(self, "Select Directory", options=k.ShowDirsOnly))
# Folder Select
# Source: https://stackoverflow.com/questions/4286036/how-to-have-a-directory-dialog
        self.folder_file_list = glob.glob(folder_path + "/*.jpg")
        self.folder_file_counter = 0
        self.folder_list_length = len(self.folder_file_list)
        self.damage_status_list = {}
        self.damage_status_list[self.folder_file_list[self.folder_file_counter]] = []
        self.display_image()
        self.set_table()
        self.prepare_display_data()
        if self.display_image_data:
            self.display_status_table()


    def display_image(self):
        myPixmap = QtGui.QPixmap(_fromUtf8(self.folder_file_list[self.folder_file_counter]))
        self.image_res = (myPixmap.size().width(), myPixmap.size().height())
        ScaledPixmap = myPixmap.scaled(self.label.size(), QtCore.Qt.KeepAspectRatio)
        self.label.setPixmap(ScaledPixmap)
        self.label_2.setText(("File Name : " + str(self.folder_file_list[self.folder_file_counter].split('/')[-1])))
        self.label_3.setText(str(self.folder_file_counter + 1) + '/' + str(self.folder_list_length))
        self.label_4.setText(("Image Resolution : " + str(myPixmap.size().width()) + '  x  ' + str(myPixmap.size().height())))
        self.item_profile = {}


    def set_table(self):
        self.tableWidget.setRowCount(0)
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(10)
        self.tableWidget.setColumnCount(2)
        self.current_row = 0


    def clear_all(self):
        file_name = self.folder_file_list[self.folder_file_counter].split('/')[-1]
        self.image_profile[file_name]['features'] = [0] * len(dataset_rows)
        self.set_table()
        self.prepare_display_data()
        self.display_status_table()


    def clear_last(self):
        if self.display_image_data:
            self.display_image_data = self.display_image_data[:-1]
            self.get_item_profile(self.display_image_data)
            self.set_table()
            self.display_status_table()
        elif self.item_profile:
            self.display_image_data = ['@'.join(i.lower().split()) + '_' + \
                                                j.lower() for i,j in \
                                                zip(list(self.item_profile.keys()), \
                                                list(self.item_profile.values()))]
            self.display_image_data = self.display_image_data[:-1]
            self.get_item_profile(self.display_image_data)
            self.set_table()
            self.display_status_table()
        else:
            pass
        


    def prepare_display_data(self):
        file_name = self.folder_file_list[self.folder_file_counter].split('/')[-1]
        try:
            self.display_image_data = self.image_profile[file_name]
            self.display_image_data = self.display_image_data['features']
            self.display_image_data = self.de_vectorize_status(self.display_image_data)
            self.get_item_profile(self.display_image_data)
        except KeyError:
            self.display_image_data = []
            pass


    def display_status_table(self):
        _translate = QtCore.QCoreApplication.translate
        for index, element in enumerate(self.display_image_data):
            item = QtWidgets.QTableWidgetItem()
            self.tableWidget.setItem(index, 0, item)
            item = QtWidgets.QTableWidgetItem()
            self.tableWidget.setItem(index, 1, item)
            item = self.tableWidget.item(index, 0)
            item.setText(_translate("SmartLabeller", dataset_rows_dict[element][0]))
            item = self.tableWidget.item(index, 1)
            item.setText(_translate("SmartLabeller", dataset_rows_dict[element][1]))
            self.current_row = index + 1


    def left_scroll(self):
        self.folder_file_counter = (self.folder_file_counter - 1) % self.folder_list_length
        self.display_image()
        self.set_table()
        self.prepare_display_data()
        self.display_status_table()


    def right_scroll(self):
        self.folder_file_counter = (self.folder_file_counter + 1) % self.folder_list_length
        self.display_image()
        self.set_table()
        self.prepare_display_data()
        self.display_status_table()


    def append_item(self):
        _translate = QtCore.QCoreApplication.translate
        col_1 = str(self.comboBox_2.currentText())
        col_2 = str(self.comboBox_3.currentText())
        self.item_profile[col_1] = '_'.join(col_2.lower().split())
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(self.current_row, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setItem(self.current_row, 1, item)
        item = self.tableWidget.item(self.current_row, 0)
        item.setText(_translate("SmartLabeller", col_1))
        item = self.tableWidget.item(self.current_row, 1)
        item.setText(_translate("SmartLabeller", col_2))
        self.current_row += 1


    def get_item_profile(self, item_data):
        self.item_profile = {dataset_rows_dict[i][0]:'_'.join(dataset_rows_dict[i][1].lower().split()) \
                             for i in item_data}


    def vectorize_status(self):
        k = ['_'.join(i.lower().split()) + '_' + j for i,j in self.item_profile.items()]
        l=[0 for i in range(len(dataset_rows))]
        for j in k:
            ind_ = dataset_rows.index(j)
            l[ind_] = 1
        return l


    def de_vectorize_status(self, vector):
        return [dataset_rows[i] for i,j in enumerate(vector) if j == 1]


    def get_image_profile(self):
        file_name = self.folder_file_list[self.folder_file_counter].split('/')[-1]
        self.image_profile[file_name] = {}
        self.image_profile[file_name]['image_name'] = file_name
        self.image_profile[file_name]['features'] = self.vectorize_status()


    def save_item(self):
        self.get_image_profile()
        with open(self.json_file_name, 'w') as outfile:
            json.dump(self.image_profile, outfile)
        time.sleep(1)
        with open(self.json_file_name) as json_file:
            self.image_profile = json.load(json_file)
        self.auto_dataset_statistics_table()


    def auto_dataset_statistics_table(self):
        k = [i for i in list(self.image_profile.values())]
        k = [list(i.values())[1] for i in k]
        k = [sum(i) for i in zip(*k)]
        dataset = {}
        dataset['Item'] = [' '.join(dataset_rows_dict[dataset_rows[i]]) for i in range(len(dataset_rows))]
        dataset['Quantity'] = [str(k[i]) for i in range(len(dataset_rows))]
        model = PandasModel(pd.DataFrame(dataset))
        self.tableView.setModel(model)
        header = self.tableView.horizontalHeader() 
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)


    def delete_image(self):
        file_name = self.folder_file_list[self.folder_file_counter].split('/')[-1]
        shutil.move(self.folder_file_list[self.folder_file_counter],
                    'dump/' + file_name)
        self.folder_file_list.pop(self.folder_file_counter)
        self.image_profile.pop(file_name, None)
        self.folder_list_length = len(self.folder_file_list)
        self.folder_file_counter = self.folder_list_length - 1 if \
                                   self.folder_file_counter >= self.folder_list_length \
                                   else self.folder_file_counter
        self.display_image()
        self.set_table()
        self.prepare_display_data()
        self.display_status_table()


    def data_split(self):
        k = list(self.image_profile.keys())
        c = self.spinBox.value()
        c = int(len(k) * c/100)
        self.train_list_names = random.sample(k, c)
        self.test_list_names = [i for i in k if i not in self.train_list_names]
        dataset = {}
        dataset['Items'] = [' '.join(dataset_rows_dict[dataset_rows[i]]) for i in range(len(dataset_rows))]
        train_list = [self.image_profile[i]['features'] for i in self.train_list_names]
        test_list = [self.image_profile[i]['features'] for i in self.test_list_names]
        train_count = [sum(i) for i in zip(*train_list)]
        test_count = [sum(i) for i in zip(*test_list)]
        dataset['Train'] = train_count
        dataset['Test'] = test_count
        overall_count = [train_count[i] + test_count[i] for i in range(len(dataset_rows))]
        t = [int((train_count[i]/overall_count[i])*100) if train_count[i] >0 else 0 for i in range(len(dataset_rows))]
        dataset['Train/Test'] = [str(i) + '/' + str(100-i) if i >0 else str(0) for i in t]
        model = PandasModel(pd.DataFrame(dataset))
        self.tableView_2.setModel(model)
        header = self.tableView_2.horizontalHeader() 
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)


    def save_split(self):
        self.train_dataset = [self.image_profile[i] for i in self.train_list_names]
        self.test_dataset = [self.image_profile[i] for i in self.test_list_names]
        self.remove_non_peril_damage()
        with open('train_dataset.pkl', 'wb') as tr:
            pickle.dump(self.train_dataset, tr, pickle.HIGHEST_PROTOCOL)
        with open('test_dataset.pkl', 'wb') as te:
            pickle.dump(self.test_dataset, te, pickle.HIGHEST_PROTOCOL)

    def remove_non_peril_damage(self):
        for i, j in enumerate(self.train_dataset):
            self.train_dataset[i]['features'] = [self.train_dataset[i]['features'][k] for k in range(len(dataset_rows)) if dataset_rows[k][-16:] != 'non_peril_damage']
        for i, j in enumerate(self.test_dataset):
            self.test_dataset[i]['features'] = [self.test_dataset[i]['features'][k] for k in range(len(dataset_rows)) if dataset_rows[k][-16:] != 'non_peril_damage']

    def remove_duplicates(self):
        pass

    def combine_json(self):
        k = self.plainTextEdit.toPlainText()
        k = k.split('\n')
        m = []
        for i in k:
            with open(i) as json_file:
                m.append(json.load(json_file))
        img_keys=[]
        for i in m:
            img_keys.extend(list(i.keys()))
        img_keys = list(set(img_keys))
        new_profile = {c:{'image_name':c,'features':[0]*24} for c in img_keys}
        for i in img_keys:
            for j in m:
                if i in j:
                    new_profile[i]['features'] = [x + y for x,y in zip(new_profile[i]['features'], j[i]['features'])]
                    new_profile[i]['features'] = [1 if l >= 1 else 0 for l in new_profile[i]['features']]
        new_file_name = self.lineEdit_2.text()
        with open(new_file_name, 'w') as outfile:
            json.dump(new_profile, outfile)


class PandasModel(QtCore.QAbstractTableModel):
    """
    Class to populate a table view with a pandas dataframe
    """
    def __init__(self, data, parent=None):
        QtCore.QAbstractTableModel.__init__(self, parent)
        self._data = data

    def rowCount(self, parent=None):
        return len(self._data.values)

    def columnCount(self, parent=None):
        return self._data.columns.size

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                return str(self._data.values[index.row()][index.column()])
        return None

    def headerData(self, x, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return self._data.columns[x]
        if orientation == QtCore.Qt.Vertical and role == QtCore.Qt.DisplayRole:
            return self._data.index[x]
        return None

def main():
    app = QApplication(sys.argv)
    form = ExampleApp()
    form.show()
    app.exec_()

if __name__ == '__main__':
    main()

