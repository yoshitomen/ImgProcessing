"""=========================================
@brief: MNISTのデータに対してlossとaccuracyの
        推移をグラフ出力する実装.
========================================="""

# ライブラリのインポート
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
from matplotlib import pyplot as plt
import datetime
import os
file_path = os.path.dirname(os.path.abspath(__file__))

def get_time_now():
    d_today = datetime.date.today()
    t_now = datetime.datetime.now()
    return d_today.strftime('%m%d') + t_now.strftime('%H%M%S')

#test用trainデータのインポートとラッチ
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data() #SSL認証必須


#Hyper parameter setting================
debug_mode = 1 #debugモード設定
learning_rate = 0.5
n_batch_size = 50
n_epochs = 20 #学習回数
input_size = X_train.shape
#=======================================

#Model Param============================
ConvFilt = [6,12]
kernel_size = [(3,3),(3,3)]
max_pooling = [(2,2),(2,2)]
#=======================================

if debug_mode: 
    print(f"Xtrain[0].shape= {X_train[0].shape}")
    print(f"X_train.shape={X_train.shape}")
    print(f"y_train.shape={y_train.shape}")
    print(f"X_test.shape={X_test.shape}")
    print(f"y_test.shape={y_test.shape}")
    
    #print(f"y_train[0].shape[0]{y_train[0].shape[0]}")
    print(f"y_train.shape[0]{y_train.shape[0]}")
    print(f"y_train{y_train}")
    #print(f"y_train[0].shape{y_train[0].shape}")
    
    for idt in range(X_test.shape[0]):
        #print(idt)
        if (X_train[0].shape[0] != X_train[idt].shape[0] and X_train[0].shape[0] != X_test[idt].shape[1]):print("ERROR1")
        if (X_train[0].shape[1] != X_train[idt].shape[1] and X_train[0].shape[1] != X_test[idt].shape[1]):print("ERROR2")
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
    file_name = f"{file_path}/result/CnnResult_{get_time_now()}.jpg"
    plt.savefig(file_name)
    print(f"SAVE FILE {file_name}")
    file_w = open(f"{file_path}/result/ParamLog_{get_time_now()}.txt",'w')
    file_w.write(f"learning_rate={learning_rate}\nbatch_size={n_batch_size}\n"\
                 +f"epochs={n_epochs}\nConv_Filt_Size={ConvFilt}\n"\
                 +f"kernel_size={kernel_size}\npooling_size={max_pooling}")
    #if debug_mode:plt.show()

# モデルの定義
model = Sequential()

#1回目のConvolution
# 1. 畳み込み層(フィルタサイズ=6, カーネルサイズ=(3,3), 入力サイズ)  
model.add(Conv2D(filters=6, kernel_size=(3, 3), input_shape=(input_size[1], input_size[2], 1)))#GrayScale=1, RGB=3

# 2. 活性化関数(Sigmoid)  
model.add(Activation("sigmoid"))
# 3. プーリング層(プーリングサイズ=(2,2))  
model.add(MaxPooling2D(pool_size=(2, 2)))

#2回目のConvolution
for idt in range(1):
    model.add(Conv2D(filters=12, kernel_size=(3, 3)))
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
# 0~9の多クラス分類なのでsparse_categorical_crossentropy. 0,1のみ(one-hot encodeing)ならcategorical_crossentropy.

# Main処理
history_log = model.fit(X_train, y_train, batch_size=n_batch_size,epochs=n_epochs,validation_data=(X_test, y_test) )

# 学習曲線の作成
learning_plot(history_log, n_epochs)