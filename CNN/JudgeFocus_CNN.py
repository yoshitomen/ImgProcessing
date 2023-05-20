#===========================================
"""
@brief: 異なるサイズの画像をリサイズして
        ラベル付きデータを自作する実装.
"""
#===========================================


# ライブラリのインポート
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
from matplotlib import pyplot as plt
import os
import cv2
import numpy as np
import datetime
import random #shuffle用

RGB = 3
GRAY_SCALE = 1

def get_time_now():
    d_today = datetime.date.today()
    t_now = datetime.datetime.now()
    return d_today.strftime('%m%d') + t_now.strftime('%H%M%S')

#PASSの設定 ※※AWS開発の場合は変更する※※=============================
DATA_DIR = "~/git-work/ManagementProject/Development/JudgeFocus_CNN/dataset/"
TRAIN_DATA_DIR = "~/git-work/ManagementProject/Development/JudgeFocus_CNN/dataset/train/"
VALID_DATA_DIR = "~/git-work/ManagementProject/Development/JudgeFocus_CNN/dataset/valid/"

#ハイパーパラメータの設定============================
debug_mode = 1 #DEBUGモード設定
learning_rate = 1
n_batch_size = 1
n_epochs = 5 #学習回数
color_num = RGB #GrayScale=1, RGB=3
#input_size = X_train.shape
data_categories = ["train", "valid"]
label_categories = ["JudgeOK", "JudgeNG"] #dataset/フォルダ名とカテゴライズ名を一致させておく.
resize_settings = (3000, 2000) #cv2の画像のリサイズ設定. 横*縦

#global変数の定義(基本的に変更しない)=================
#training_data = []
#alidation_data = [] #np.ones((2,2))
#image_original_train_array = []
#image_resize_train_array = []
#image_original_valid_array = [] #np.zeros(3)
#image_resize_valid_array = []
X_train=[]
y_train=[]
X_test=[]
y_test=[]
file_name = f"valid_process/JfCnn_{get_time_now()}.jpg" # ※実行場所から見たパス設定注意
#model = Sequential()
#====================================================

#test用trainデータのインポートとラッチ
"""
from keras.datasets import mnist
(chkX_train, chky_train), (chkX_test, chky_test) = mnist.load_data() #これはnumpy arrayぽい
input_size = chkX_train.shape
print(chkX_train.shape)
print(chky_train.shape)
"""


    

#画像データの読み込みと下処理(教師/検証、ラベル付け)
def CreateDataset():
    #スコープ変数の定義
    training_data = []
    validation_data = []
    
    #まずtrainデータとvalidデータで分岐
    for label_tr_va, data_type in enumerate(data_categories):
        
        #JudgeOK,JudgeNGで分岐.
        for label_num, category in enumerate(label_categories):
            category_folder_path = DATA_DIR+str(data_type)+"/"
            #print(category_folder_path)
            category_folder_path = os.path.join(category_folder_path, category)
            #print(category_folder_path)
            
            #フォルダ内にファイルがない場合はエラー文を出力.
            if os.listdir(category_folder_path)==[]: print(f"NO FILE in DIR_{category_folder_path}")
            
            #それぞれのフォルダ内の画像を読み込んで合否を示すenumをつける.
            for image in os.listdir(category_folder_path): #画像を走査する.
                #ここに入った時点でフォルダとしては最下層(train_ok-valid_ngまでの4つのどれか)
                if(data_type == "train"):
                    if debug_mode:
                        print("train:for")
                    print(image)
                    image_original_train_array = cv2.imread(os.path.join(category_folder_path, image),)
                    image_resize_train_array = cv2.resize(image_original_train_array, resize_settings)
                    training_data.append([image_resize_train_array, label_num])
                                        
                elif(data_type == "valid"):
                    if debug_mode:
                        print("valid:for")
                        print(image)
                    image_original_valid_array = cv2.imread(os.path.join(category_folder_path, image),)
                    image_resize_valid_array = cv2.resize(image_original_valid_array, resize_settings)
                    validation_data.append([image_resize_valid_array, label_num])
                else: print("DATA SET ERROR")
    if debug_mode:print(f"TrainNum={len(training_data)} \nValidNum={len(validation_data)}")
    random.shuffle(training_data)
    random.shuffle(validation_data)
    
    global X_train,y_train,X_test,y_test
    for feature, label in training_data:
        X_train.append(feature)
        y_train.append(label)
    for feature, label in validation_data:
        X_test.append(feature)
        y_test.append(label)
    
    #print(training_data[0])
    #print(training_data[1])
    
    X_train = np.array(X_train) #shape=(OK+NG枚数, 縦pixel, 横pixel, color_channel)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    if (X_train.shape[0] != y_train.shape[0]):print("ERROR3")

#学習曲線の作成
def learning_plot(history, epochs):
    fig = plt.figure(figsize=(11,4))
    # Lossの可視化
    plt.subplot(1,2,1)
    plt.plot(range(1,epochs+1), history.history['loss'])
    plt.plot(range(1,epochs+1), history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.xticks(range(1,epochs+1))
    plt.ylabel('loss')
    plt.legend(['training', 'validation'], loc='upper right')
    
    # 正解率(accuracy)の可視化
    plt.subplot(1,2,2)
    plt.plot(range(1,epochs+1), history.history['accuracy'])
    plt.plot(range(1,epochs+1), history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.xticks(range(1,epochs+1))
    plt.ylabel('accuracy')
    plt.legend(['training', 'validation'], loc='lower right')
    #file_name = f"valid_process/JfCnn_{get_time_now()}.jpg"
    plt.savefig(file_name)
    #print(f"SAVE FILE {file_name}")
    #plt.show()

# モデルの定義
model = Sequential()

#1回目のConvolution
# 1. 畳み込み層(フィルタサイズ=6, カーネルサイズ=(3,3), 入力サイズinput_shape=(縦横(※resizeは横縦)、RGB or Gray))  
model.add(Conv2D(filters=6, kernel_size=(3, 3), input_shape=(resize_settings[1], resize_settings[0], color_num)))

# 2. 活性化関数(Sigmoid)  
model.add(Activation("sigmoid"))
# 3. プーリング層(プーリングサイズ=(2,2))  
model.add(MaxPooling2D(pool_size=(2, 2)))

#2回目のConvolution
for idt in range(1):
    model.add(Conv2D(filters=6, kernel_size=(3, 3)))
    model.add(Activation("sigmoid"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(filters=24, kernel_size=(3, 3)))
# model.add(Activation("sigmoid"))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# 7. 平坦化(). 1次元テンソルにする
model.add(Flatten())

# ここからNN
# 8. 全結合層(出力=120)  
model.add(Dense(units=120))
# 9. 活性化関数(Sigmoid)  
model.add(Activation("sigmoid"))

model.add(Dense(units=80))
model.add(Activation("sigmoid"))

# 最後の出力は合計を1となる分布にするので活性化関数はSoftmax.
model.add(Dense(units=10))
model.add(Activation("softmax"))

# モデルの確認
model.summary()

# 損失関数、オプティマイザ、学習率(learning rate)、評価関数(metrics)を指定.
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizers.SGD(lr=learning_rate), metrics=["accuracy"])
# one-hot encodingなのでcategorical_crossentropy

# Main処理
CreateDataset()
history_log = model.fit(X_train, y_train, batch_size=n_batch_size,epochs=n_epochs,validation_data=(X_test, y_test) )

# 学習曲線の作成
learning_plot(history_log, n_epochs)