#@title РЈСЃС‚Р°РЅРѕРІРєР° РјРѕРґСѓР»СЏ РЈРР
from PIL import Image
from pathlib import Path
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from IPython import display as ipd

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import seaborn as sns 

import gdown
import zipfile
import os
import random
import time 
import gc

sns.set(style='darkgrid') 
seed_value = 12
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class AccuracyCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.train_acc = []
        self.val_acc = []
        self.train_loss = []
        self.val_loss = []
        self.times = []


    def plot_graph(self):        
        plt.figure(figsize=(20, 14))
        plt.subplot(2, 2, 1)
        plt.title('РўРѕС‡РЅРѕСЃС‚СЊ', fontweight='bold')
        plt.plot(self.train_acc, label='РўРѕС‡РЅРѕСЃС‚СЊ РЅР° РѕР±СѓС‡Р°С‰РµР№ РІС‹Р±РѕСЂРєРµ')
        plt.plot(self.val_acc, label='РўРѕС‡РЅРѕСЃС‚СЊ РЅР° РїСЂРѕРІРµСЂРѕС‡РЅРѕР№ РІС‹Р±РѕСЂРєРµ')
        plt.xlabel('Р­РїРѕС…Р° РѕР±СѓС‡РµРЅРёСЏ')
        plt.ylabel('Р”РѕР»СЏ РІРµСЂРЅС‹С… РѕС‚РІРµС‚РѕРІ')
        plt.legend()        
        plt.show()
       

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.train_acc.append(logs['accuracy'])
        self.val_acc.append(logs['val_accuracy'])
        self.train_loss.append(logs['loss'])
        self.val_loss.append(logs['val_loss'])
        t = round(time.time() - self.start_time, 1)
        self.times.append(t)
        if logs['val_accuracy'] > self.accuracymax:
            self.accuracymax = logs['val_accuracy']
            self.idxmax = epoch
        print(f'Р­РїРѕС…Р° {epoch+1}'.ljust(10)+ f'Р’СЂРµРјСЏ РѕР±СѓС‡РµРЅРёСЏ: {t}c'.ljust(25) + f'РўРѕС‡РЅРѕСЃС‚СЊ РЅР° РѕР±СѓС‡Р°СЋС‰РµР№ РІС‹Р±РѕСЂРєРµ: {bcolors.OKBLUE}{round(logs["accuracy"]*100,1)}%{bcolors.ENDC}'.ljust(50) +f'РўРѕС‡РЅРѕСЃС‚СЊ РЅР° РїСЂРѕРІРµСЂРѕС‡РЅРѕР№ РІС‹Р±РѕСЂРєРµ: {bcolors.OKBLUE}{round(logs["val_accuracy"]*100,1)}%{bcolors.ENDC}')
        self.cntepochs += 1

    def on_train_begin(self, logs):
        self.idxmax = 0
        self.accuracymax = 0
        self.cntepochs = 0

    def on_train_end(self, logs):
        ipd.clear_output(wait=True)
        for i in range(self.cntepochs):
            if i == self.idxmax:
                print('\33[102m' + f'Р­РїРѕС…Р° {i+1}'.ljust(10)+ f'Р’СЂРµРјСЏ РѕР±СѓС‡РµРЅРёСЏ: {self.times[i]}c'.ljust(25) + f'РўРѕС‡РЅРѕСЃС‚СЊ РЅР° РѕР±СѓС‡Р°СЋС‰РµР№ РІС‹Р±РѕСЂРєРµ: {round(self.train_acc[i]*100,1)}%'.ljust(41) +f'РўРѕС‡РЅРѕСЃС‚СЊ РЅР° РїСЂРѕРІРµСЂРѕС‡РЅРѕР№ РІС‹Р±РѕСЂРєРµ: {round(self.val_acc[i]*100,1)}%'+ '\033[0m')
            else:
                print(f'Р­РїРѕС…Р° {i+1}'.ljust(10)+ f'Р’СЂРµРјСЏ РѕР±СѓС‡РµРЅРёСЏ: {self.times[i]}c'.ljust(25) + f'РўРѕС‡РЅРѕСЃС‚СЊ РЅР° РѕР±СѓС‡Р°СЋС‰РµР№ РІС‹Р±РѕСЂРєРµ: {bcolors.OKBLUE}{round(self.train_acc[i]*100,1)}%{bcolors.ENDC}'.ljust(50) +f'РўРѕС‡РЅРѕСЃС‚СЊ РЅР° РїСЂРѕРІРµСЂРѕС‡РЅРѕР№ РІС‹Р±РѕСЂРєРµ: {bcolors.OKBLUE}{round(self.val_acc[i]*100,1)}%{bcolors.ENDC}' )
        self.plot_graph()

class TerraDataset:
    bases = {
        'РњРѕР»РѕС‡РЅР°СЏ_РїСЂРѕРґСѓРєС†РёСЏ' : {
            'url': 'https://storage.yandexcloud.net/terraai/sources/milk.zip',
            'info': 'Р’С‹ СЃРєР°С‡Р°Р»Рё Р±Р°Р·Сѓ СЃ РёР·РѕР±СЂР°Р¶РµРЅРёСЏРјРё Р±СѓС‚С‹Р»РѕРє РјРѕР»РѕРєР°. Р‘Р°Р·Р° СЃРѕРґРµСЂР¶РёС‚ 1500 РёР·РѕР±СЂР°Р¶РµРЅРёР№ С‚СЂРµС… РєР°С‚РµРіРѕСЂРёР№: В«ParmalatВ», В«РљСѓР±Р°РЅСЃРєР°СЏ Р±СѓСЂРµРЅРєР°В», В«РЎРµРјРµР№РЅС‹Р№ С„РѕСЂРјР°С‚В»',
            'dir_name': 'milk_ds',
            'task_type': 'img_classification',
            'size': (96, 53),
        },
        'РџР°СЃСЃР°Р¶РёСЂС‹_Р°РІС‚РѕР±СѓСЃР°' : {
            'url': 'https://storage.yandexcloud.net/terraai/sources/bus.zip',
            'info': 'Р’С‹ СЃРєР°С‡Р°Р»Рё Р±Р°Р·Сѓ СЃ РёР·РѕР±СЂР°Р¶РµРЅРёСЏРјРё РїР°СЃСЃР°Р¶РёСЂРѕРІ Р°РІС‚РѕР±СѓСЃР°. Р‘Р°Р·Р° СЃРѕРґРµСЂР¶РёС‚ 9081 РёР·РѕР±СЂР°Р¶РµРЅРёРµ РґРІСѓС… РєР°С‚РµРіРѕСЂРёР№: В«Р’С…РѕРґСЏС‰РёРµ РїР°СЃСЃР°Р¶РёСЂС‹В», В«Р’С‹С…РѕРґСЏС‰РёРµ РїР°СЃР°Р¶РёСЂС‹В»',
            'dir_name': 'passengers',
            'task_type': 'img_classification',
            'size': (128, 64),
        },
        'Р’РѕР·РіРѕСЂР°РЅРёСЏ' : {
            'url': 'https://storage.yandexcloud.net/terraai/sources/fire.zip',
            'info': 'Р’С‹ СЃРєР°С‡Р°Р»Рё Р±Р°Р·Сѓ СЃ РёР·РѕР±СЂР°Р¶РµРЅРёСЏРјРё РІРѕР·РіРѕСЂР°РЅРёР№. Р‘Р°Р·Р° СЃРѕРґРµСЂР¶РёС‚ 6438 РёР·РѕР±СЂР°Р¶РµРЅРёРµ РґРІСѓС… РєР°С‚РµРіРѕСЂРёР№: В«Р•СЃС‚СЊ РІРѕР·РіРѕСЂР°РЅРёРµВ», В«РќРµС‚ РІРѕР·РіРѕСЂР°РЅРёСЏВ»',
            'dir_name': 'fire',
            'task_type': 'img_classification',
            'size': (96, 76),
        },
        'Р°РІС‚Рѕ' : {
            'url': 'https://storage.yandexcloud.net/aiueducation/Intensive/cars.zip',
            'info': 'Р’С‹ СЃРєР°С‡Р°Р»Рё Р±Р°Р·Сѓ СЃ РёР·РѕР±СЂР°Р¶РµРЅРёСЏРјРё РјР°СЂРѕРє Р°РІС‚Рѕ. Р‘Р°Р·Р° СЃРѕРґРµСЂР¶РёС‚ 3427 РёР·РѕР±СЂР°Р¶РµРЅРёР№ С‚СЂРµС… РєР°С‚РµРіРѕСЂРёР№: В«Р¤РµСЂСЂР°СЂРёВ», В«РњРµСЂСЃРµРґРµСЃВ», В«Р РµРЅРѕВ»',
            'dir_name': 'car',
            'task_type': 'img_classification',
            'size': (54, 96),
        },
        'РјР°Р№РѕРЅРµР·' : {
            'url': 'https://storage.yandexcloud.net/aiueducation/Intensive/mayonnaise.zip',
            'info': 'Р’С‹ СЃРєР°С‡Р°Р»Рё Р±Р°Р·Сѓ СЃ РёР·РѕР±СЂР°Р¶РµРЅРёСЏРјРё Р±СЂРµРЅРґРѕРІ РјР°Р№РѕРЅРµР·Р°. Р‘Р°Р·Р° СЃРѕРґРµСЂР¶РёС‚ 150 РёР·РѕР±СЂР°Р¶РµРЅРёР№ С‚СЂРµС… РєР°С‚РµРіРѕСЂРёР№: В«Р•Р–РљВ», В«РњР°С…РµРµРІВ», В«Р СЏР±Р°В»',
            'dir_name': 'mayonesse',
            'task_type': 'img_classification',
            'size': (96, 76),
        },
    }
    def __init__(self, name):
        '''
        parameters:
            name - РЅР°Р·РІР°РЅРёРµ РґР°С‚Р°СЃРµС‚Р°
        '''        
        self.base = self.bases[name]
        self.sets = None
        self.classes = None

    def load(self):
        '''
        С„СѓРЅРєС†РёСЏ Р·Р°РіСЂСѓР·РєРё РґР°С‚Р°СЃРµС‚Р°
        '''
        
        print(f'{bcolors.BOLD}Р—Р°РіСЂСѓР·РєР° РґР°С‚Р°СЃРµС‚Р°{bcolors.ENDC}',end=' ')
        
        # Р—Р°РіСѓСЂР·РєР° РґР°С‚Р°СЃРµС‚Р° РёР· РѕР±Р»Р°РєР°
        fname = gdown.download(self.base['url'], None, quiet=True)

        if Path(fname).suffix == '.zip':
            # Р Р°СЃРїР°РєРѕРІРєР° Р°СЂС…РёРІР°
            with zipfile.ZipFile(fname, 'r') as zip_ref:
                zip_ref.extractall(self.base['dir_name'])

            # РЈРґР°Р»РµРЅРёРµ Р°СЂС…РёРІР°
            os.remove(fname)

        # Р’С‹РІРѕРґ РёРЅС„РѕСЂРјР°С†РёРѕРЅРЅРѕРіРѕ Р±Р»РѕРєР°
        print(f'{bcolors.OKGREEN}Ok{bcolors.ENDC}')
        print(f'{bcolors.OKBLUE}РРЅС„Рѕ:{bcolors.ENDC}')
        print(f'    {self.base["info"]}')
        return self.base['task_type']

    def samples(self):
        '''
        Р¤СѓРЅРєС†РёСЏ РІРёР·СѓР°Р»РёР·Р°С†РёРё РїСЂРёРјРµСЂРѕРІ
        '''
        
        # Р’РёР·СѓР°Р»РёР·Р°С†РёСЏ РґР°С‚Р°СЃРµС‚Р° РёР·РѕР±СЂР°Р¶РµРЅРёР№ РґР»СЏ Р·Р°РґР°С‡Рё РєР»Р°СЃСЃРёС„РёРєР°С†РёРё
        if self.base['task_type'] == 'img_classification':
            # РџРѕР»СѓС‡РµРЅРёРµ СЃРїРёСЃРєР° РєР»Р°СЃСЃРѕРІ (РЅР°Р·РІР°РЅРёСЏ РїР°РїРѕРє РІ РґРёСЂРµРєС‚РѕСЂРёРё)
            self.classes = sorted(os.listdir(self.base['dir_name']))

            # РџРѕСЃС‚СЂРѕРµРЅРёРµ РїРѕР»РѕС‚РЅР°СЏ РІРёР·СѓР°Р»РёР·Р°С†РёРё
            f, ax = plt.subplots(len(self.classes), 5, figsize=(24, len(self.classes) * 4))
            for i, class_ in enumerate(self.classes):
                # Р’С‹Р±РѕСЂ СЃР»СѓС‡Р°Р№РЅРѕРіРѕ РёР·РѕР±СЂР°Р¶РµРЅРёСЏ
                for j in range(5):
                  random_image = random.choice(
                      os.listdir(os.path.join(
                          self.base['dir_name'], 
                          class_)))
                  img = Image.open(os.path.join(
                      self.base['dir_name'],
                      class_,
                      random_image))
                  ax[i, j].imshow(img)
                  ax[i, j].axis('off')
                  ax[i, j].set_title(class_)
            plt.show()   

    def create_sets(self):
        '''
        Р¤СѓРЅРєС†РёСЏ СЃРѕР·РґР°РЅРёСЏ РІС‹Р±РѕСЂРѕРє
        '''
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        print(f'{bcolors.BOLD}РЎРѕР·РґР°РЅРёРµ РЅР°Р±РѕСЂРѕРІ РґР°РЅРЅС‹С… РґР»СЏ РѕР±СѓС‡РµРЅРёСЏ РјРѕРґРµР»Рё{bcolors.ENDC}', end=' ')

        # РЎРѕР·РґР°РЅРёРµ РІС‹Р±РѕСЂРѕРє РґР»СЏ Р·Р°РґР°С‡Рё РєР»Р°СЃСЃРёС„РёРєР°С†РёРё РёР·РѕР±СЂР°Р¶РµРЅРёР№
        if self.base['task_type'] == 'img_classification':

            # РџРѕР»СѓС‡РµРЅРёРµ СЃРїРёСЃРєР° РґРёСЂРµРєС‚РѕСЂРёР№
            self.classes = sorted(os.listdir(self.base['dir_name']))
            counts = []

            # РџСЂРѕС…РѕРґ РїРѕ РІСЃРµРј РїР°РїРєР°Рј РґРёСЂРµРєС‚РѕСЂРёРё (РїРѕ РІСЃРµРј РєР»Р°СЃСЃР°Рј)
            for j, d in enumerate(self.classes):

              # РџРѕР»СѓС‡РµРЅРёРµ СЃРїРёСЃРєР° РІСЃРµС… РёР·РѕР±СЂР°Р¶РµРЅРёР№ РѕС‡РµСЂРµРґРЅРѕРіРѕ РєР»Р°СЃСЃР°
              files = sorted(os.listdir(os.path.join(self.base['dir_name'], d)))

              # РџР°СЂР°РјРµС‚СЂ СЂР°Р·РґРµР»РµРЅРёСЏ РІС‹Р±РѕСЂРѕРє
              counts.append(len(files))
              count = counts[-1] * .9

              # РџСЂРѕС…РѕРґ РїРѕ РІСЃРµРј РёР·РѕР±СЂР°Р¶РµРЅРёСЏРј РѕС‡РµСЂРµРґРЅРѕРіРѕ РєР»Р°СЃСЃР°
              for i in range(len(files)):
                  
                  # Р—Р°РіСЂСѓР·РєР° РѕС‡РµСЂРµРґРЅРѕРіРѕ РёР·РѕР±СЂР°Р¶РµРЅРёСЏ
                  sample = np.array(image.load_img(os.path.join(
                      self.base['dir_name'],
                      d,
                      files[i]), target_size=self.base['size']))
                  
                  # Р”РѕР±Р°РІР»РµРЅРёРµ СЌР»РµРјРµРЅС‚Р° РІ С‚РµСЃС‚РѕРІСѓСЋ РёР»Рё РїСЂРѕРІРµСЂРѕС‡РЅСѓСЋ РІС‹Р±РѕСЂРєСѓ
                  if i<count:
                    x_train.append(sample)
                    y_train.append(j)
                  else:
                    x_test.append(sample)
                    y_test.append(j)
            self.sets = (np.array(x_train)/255., np.array(y_train)), (np.array(x_test)/255., np.array(y_test))

            # Р’С‹РІРѕРґ С„РёРЅР°Р»СЊРЅРѕР№ РёРЅС„РѕСЂРјР°С†РёРё
            print(f'{bcolors.OKGREEN}Ok{bcolors.ENDC}')
            print()
            print(f'Р Р°Р·РјРµСЂ СЃРѕР·РґР°РЅРЅС‹С… РІС‹Р±РѕСЂРѕРє:')
            print(f'  РћР±СѓС‡Р°СЋС‰Р°СЏ РІС‹Р±РѕСЂРєР°: {self.sets[0][0].shape}')
            print(f'  РњРµС‚РєРё РѕР±СѓС‡Р°СЋС‰РµР№ РІС‹Р±РѕСЂРєРё: {self.sets[0][1].shape}')
            print(f'  РџСЂРѕРІРµСЂРѕС‡РЅР°СЏ РІС‹Р±РѕСЂРєР°: {self.sets[1][0].shape}')
            print(f'  РњРµС‚РєРё РїСЂРѕРІРµСЂРѕС‡РЅРѕР№ РІС‹Р±РѕСЂРєРё: {self.sets[1][1].shape}')
            print()
            print(f'Р Р°СЃРїСЂРµРґРµР»РµРЅРёРµ РїРѕ РєР»Р°СЃСЃР°Рј:')
            f, ax =plt.subplots(1,2, figsize=(16, 5))            
            ax[0].bar(self.classes, np.array(counts)*0.9)
            ax[0].set_title('РћР±СѓС‡Р°СЋС‰Р°СЏ РІС‹Р±РѕСЂРєР°')
            ax[1].bar(self.classes, np.array(counts)*0.1, color='g')
            ax[1].set_title('РџСЂРѕРІРµСЂРѕС‡РЅР°СЏ РІС‹Р±РѕСЂРєР°')
            plt.show()


class TerraModel:
    def __init__(self, task_type, trds):
        self.model = None
        self.task_type = task_type
        self.trds = trds
    
    @staticmethod
    def create_layer(params):
        '''
           Р¤СѓРЅРєС†РёСЏ СЃРѕР·РґР°РЅРёСЏ СЃР»РѕСЏ
        '''
        activation = 'relu'
        params = params.split('-')
        
        # Р”РѕР±Р°РІР»РµРЅРёРµ РІС…РѕРґРЅРѕРіРѕ СЃР»РѕСЏ
        if params[0].lower() == 'РІС…РѕРґРЅРѕР№':
            return Input(shape=eval(params[1]))

        # Р”РѕР±Р°РІР»РµРЅРёРµ РїРѕР»РЅРѕСЃРІСЏР·РЅРѕРіРѕ СЃР»РѕСЏ
        if params[0].lower() == 'РїРѕР»РЅРѕСЃРІСЏР·РЅС‹Р№':
            if len(params)>2:
                activation = params[2]
            return Dense(eval(params[1]), activation=activation)

        # Р”РѕР±Р°РІР»РµРЅРёРµ РІС‹СЂР°РІРЅРёРІР°СЋС‰РµРіРѕ СЃР»РѕСЏ
        if params[0].lower() == 'РІС‹СЂР°РІРЅРёРІР°СЋС‰РёР№':
            return Flatten()

        # Р”РѕР±Р°РІР»РµРЅРёРµ СЃРІРµСЂС‚РѕС‡РЅРѕРіРѕ СЃР»РѕСЏ (Conv2D)
        if params[0].lower() == 'СЃРІРµСЂС‚РѕС‡РЅС‹Р№2Рґ':
            if len(params)>3:
                activation = params[3]
            return Conv2D(eval(params[1]), eval(params[2]), activation=activation, padding='same')
            
    def create_model(self, layers):
        '''
        Р¤СѓРЅРєС†РёСЏ СЃРѕР·РґР°РЅРёСЏ РЅРµР№СЂРѕРЅРЅРѕР№ СЃРµС‚Рё
        parameters:
            layers - СЃР»РѕРё (С‚РµРєСЃС‚РѕРј)
        '''
        if self.task_type=='img_classification':
            layers += '-softmax'        
        layers = layers.split()
        # РЎРѕР·РґР°РЅРёРµ РІС…РѕРґРЅРѕРіРѕ СЃР»РѕСЏ
        inp = self.create_layer(f'РІС…РѕРґРЅРѕР№-{self.trds.sets[0][0].shape[1:]}')

        # РЎРѕР·РґР°РЅРёРµ РїРµСЂРІРѕРіРѕ СЃР»РѕСЏ
        x = self.create_layer(layers[0]) (inp)

        # РЎРѕР·РґР°РЅРёРµ РѕСЃС‚Р°Р»СЊРЅС‹С… СЃР»РѕРµРІ
        for layer in layers[1:]:
            x = self.create_layer(layer) (x)            
        self.model = Model(inp, x)        

    def train_model(self, epochs, use_callback=True):
        '''
        Р¤СѓРЅРєС†РёСЏ РѕР±СѓС‡РµРЅРёСЏ РЅРµР№СЂРѕРЅРЅРѕР№ СЃРµС‚Рё
        parameters:
            epochs - РєРѕР»РёС‡РµСЃС‚РІРѕ СЌРїРѕС…
        '''
        
        # РћР±СѓС‡РµРЅРёРµ РјРѕРґРµР»Рё РєР»Р°СЃСЃРёС„РёРєР°С†РёРё РёР·РѕР±СЂР°Р¶РµРЅРёР№
        if self.task_type=='img_classification':
            self.model.compile(loss='sparse_categorical_crossentropy', optimizer = Adam(0.0001), metrics=['accuracy'])
            accuracy_callback = AccuracyCallback()
            callbacks = []
            if use_callback:
                callbacks = [accuracy_callback]
            history = self.model.fit(self.trds.sets[0][0], self.trds.sets[0][1],
                          batch_size = self.trds.sets[0][0].shape[0]//25,
                          validation_data=(self.trds.sets[1][0], self.trds.sets[1][1]),
                          epochs=epochs,
                          callbacks=callbacks,
                          verbose = 0)
            return history
            
    def test_model(self):
        '''
        Р¤СѓРЅРєС†РёСЏ С‚РµСЃС‚РёСЂРѕРІР°РЅРёСЏ РјРѕРґРµР»Рё
        '''
        # РўРµСЃС‚РёСЂРѕРІР°РЅРёРµ РјРѕРґРµР»Рё РєР»Р°СЃСЃРёС„РёРєР°С†РёРё РёР·РѕР±СЂР°Р¶РµРЅРёР№
        if self.task_type=='img_classification':
            for i in range(10):
                number = np.random.randint(self.trds.sets[1][0].shape[0])
                sample = self.trds.sets[1][0][number]
                print('РўРµСЃС‚РѕРІРѕРµ РёР·РѕР±СЂР°Р¶РµРЅРёРµ:')
                plt.imshow(sample) # Р’С‹РІРѕРґРёРј РёР·РѕР±СЂР°Р¶РµРЅРёРµ РёР· С‚РµСЃС‚РѕРІРѕРіРѕ РЅР°Р±РѕСЂР° СЃ Р·Р°РґР°РЅРЅС‹Рј РёРЅРґРµРєСЃРѕРј
                plt.axis('off') # РћС‚РєР»СЋС‡Р°РµРј РѕСЃРё
                plt.show() 
                pred = self.model.predict(sample[None, ...])[0]
                max_idx = np.argmax(pred)
                print()
                print('Р РµР·СѓР»СЊС‚Р°С‚ РїСЂРµРґСЃРєР°Р·Р°РЅРёСЏ РјРѕРґРµР»Рё:')
                for i in range(len(self.trds.classes)):
                    if i == max_idx:
                        print(bcolors.BOLD, end='')
                    print(f'РњРѕРґРµР»СЊ СЂР°СЃРїРѕР·РЅР°Р»Р° РєР»Р°СЃСЃ В«{self.trds.classes[i]}В» РЅР° {round(100*pred[i],1)}%{bcolors.ENDC}')
                print('---------------------------')
                print('РџСЂР°РІРёР»СЊРЅС‹Р№ РѕС‚РІРµС‚: ',end='')
                if max_idx == self.trds.sets[1][1][number]:
                    print(bcolors.OKGREEN, end='')
                else:
                    print(bcolors.FAIL, end='')
                print(self.trds.classes[self.trds.sets[1][1][number]],end=f'{bcolors.ENDC}\n')
                print('---------------------------')
                print()
                print()


class TerraIntensive:
    def __init__(self):
       self.trds = None
       self.trmodel = None
       self.task_type = None

    def load_dataset(self, ds_name):
        self.trds = TerraDataset(ds_name)
        self.task_type = self.trds.load()

    def samples(self):
        self.trds.samples()

    def create_sets(self):
        self.trds.create_sets()

    def create_model(self, layers):
        print(f'{bcolors.BOLD}РЎРѕР·РґР°РЅРёРµ РјРѕРґРµР»Рё РЅРµР№СЂРѕРЅРЅРѕР№ СЃРµС‚Рё{bcolors.ENDC}', end=' ')
        self.trmodel = TerraModel(self.task_type, self.trds)
        self.trmodel.create_model(layers)
        print(f'{bcolors.OKGREEN}Ok{bcolors.ENDC}')

    def train_model(self, epochs):
        self.trmodel.train_model(epochs)

    def test_model(self):
        self.trmodel.test_model()

    def train_model_average(self, layers, cnt=10):
        if self.task_type == 'img_classification':
          print(f'{bcolors.BOLD}РћРїСЂРµРґРµР»РµРЅРёРµ СЃСЂРµРґРЅРµРіРѕ РїРѕРєР°Р·Р°С‚РµР»СЏ С‚РѕС‡РЅРѕСЃС‚Рё РјРѕРґРµР»Рё РЅР° {cnt} Р·Р°РїСѓСЃРєР°С…{bcolors.ENDC}')
          print()
          average_accuracy = []
          average_val_accuracy = []
          times=[]
          for i in range(cnt):
              start_time = time.time()
              self.trmodel.create_model(layers)
              history = self.trmodel.train_model(20, False).history
              average_accuracy.append(np.max(history['accuracy']))
              average_val_accuracy.append(np.max(history['val_accuracy']))
              t = round(time.time() - start_time, 1)
              times.append(t)
              print(f'Р—Р°РїСѓСЃРє {i+1}'.ljust(10)+ f'Р’СЂРµРјСЏ РѕР±СѓС‡РµРЅРёСЏ: {t}c'.ljust(25) + f'РўРѕС‡РЅРѕСЃС‚СЊ РЅР° РѕР±СѓС‡Р°СЋС‰РµР№ РІС‹Р±РѕСЂРєРµ: {bcolors.OKBLUE}{round(average_accuracy[-1]*100,1)}%{bcolors.ENDC}'.ljust(50) +f'РўРѕС‡РЅРѕСЃС‚СЊ РЅР° РїСЂРѕРІРµСЂРѕС‡РЅРѕР№ РІС‹Р±РѕСЂРєРµ: {bcolors.OKBLUE}{round(average_val_accuracy[-1]*100,1)}%{bcolors.ENDC}')
              gc.collect()
               
          ipd.clear_output(wait=True)
          print(f'{bcolors.BOLD}РћРїСЂРµРґРµР»РµРЅРёРµ СЃСЂРµРґРЅРµРіРѕ РїРѕРєР°Р·Р°С‚РµР»СЏ С‚РѕС‡РЅРѕСЃС‚Рё РјРѕРґРµР»Рё РЅР° {cnt} Р·Р°РїСѓСЃРєР°С…{bcolors.ENDC}')
          print()
          argmax_idx = np.argmax(average_val_accuracy)
          for i in range(cnt):
              if i == argmax_idx:
                  print('\33[102m' + f'Р—Р°РїСѓСЃРє {i+1}'.ljust(10)+ f'Р’СЂРµРјСЏ РѕР±СѓС‡РµРЅРёСЏ: {times[i]}c'.ljust(25) + f'РўРѕС‡РЅРѕСЃС‚СЊ РЅР° РѕР±СѓС‡Р°СЋС‰РµР№ РІС‹Р±РѕСЂРєРµ: {round(average_accuracy[i]*100,1)}%'.ljust(41) +f'РўРѕС‡РЅРѕСЃС‚СЊ РЅР° РїСЂРѕРІРµСЂРѕС‡РЅРѕР№ РІС‹Р±РѕСЂРєРµ: {round(average_val_accuracy[i]*100,1)}%'+ '\033[0m')
              else:
                  print(f'Р—Р°РїСѓСЃРє {i+1}'.ljust(10)+ f'Р’СЂРµРјСЏ РѕР±СѓС‡РµРЅРёСЏ: {times[i]}c'.ljust(25) + f'РўРѕС‡РЅРѕСЃС‚СЊ РЅР° РѕР±СѓС‡Р°СЋС‰РµР№ РІС‹Р±РѕСЂРєРµ: {bcolors.OKBLUE}{round(average_accuracy[i]*100,1)}%{bcolors.ENDC}'.ljust(50) +f'РўРѕС‡РЅРѕСЃС‚СЊ РЅР° РїСЂРѕРІРµСЂРѕС‡РЅРѕР№ РІС‹Р±РѕСЂРєРµ: {bcolors.OKBLUE}{round(average_val_accuracy[i]*100,1)}%{bcolors.ENDC}' )
          print()
          print(f'{bcolors.BOLD}РЎСЂРµРґРЅСЏСЏ С‚РѕС‡РЅРѕСЃС‚СЊ РЅР° РѕР±СѓС‡Р°СЋС‰РµР№ РІС‹Р±РѕСЂРєРµ: {bcolors.ENDC}{round(np.mean(average_accuracy[i])*100,1)}%')
          print(f'{bcolors.BOLD}РњР°РєСЃРёРјР°Р»СЊРЅР°СЏ С‚РѕС‡РЅРѕСЃС‚СЊ РЅР° РѕР±СѓС‡Р°СЋС‰РµР№ РІС‹Р±РѕСЂРєРµ: {bcolors.ENDC}{round(np.max(average_accuracy[i])*100,1)}%')
          print(f'{bcolors.BOLD}РЎСЂРµРґРЅСЏСЏ С‚РѕС‡РЅРѕСЃС‚СЊ РЅР° РїСЂРѕРІРµСЂРѕС‡РЅРѕР№ РІС‹Р±РѕСЂРєРµ: {round(np.mean(average_val_accuracy[i])*100,1)}%')
          print(f'{bcolors.BOLD}РњР°РєСЃРёРјР°Р»СЊРЅР°СЏ С‚РѕС‡РЅРѕСЃС‚СЊ РЅР° РїСЂРѕРІРµСЂРѕС‡РЅРѕР№ РІС‹Р±РѕСЂРєРµ: {round(np.max(average_val_accuracy[i])*100,1)}%')


terra_ai = TerraIntensive()