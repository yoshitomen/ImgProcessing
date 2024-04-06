import tensorflow as tf
import numpy as np
import math

from tensorflow.keras import optimizers
from tensorflow.keras.datasets import mnist
#output = activation(dot(W, input) + b) これはここで定義？
# これは教科書での説明だけ.
# 使っているところを見るとactivation=tf.nn.reluとなっている.
# 実際にはReLu関数の引数にAffine変換の結果を渡してるだけ. ReLu(Affine(output))

#Denseクラス Dense層:密結合されたNN層(全結合層)
class NaiveDense:
    def __init__(self, input_size, output_size, activation):
        self.activation = activation
        
        #形が(input_size, output_size)で初期値乱数の行列W
        w_shape = (input_size, output_size)
        w_initial_value = tf.random.uniform(shape=w_shape, minval=0, maxval=1e-1)
        #tf.random.uniformで検索したら定義出てくる
        self.W = tf.Variable(w_initial_value)
        # Variableクラス：変数の外観と動作はテンソルに似ており、実際にデータ構造が tf.Tensor で裏付け
        # テンソルのように dtype と形状を持ち、NumPy にエクスポートできます。

        #形状が(output_size,)、初期値が0のベクトルbを作成
        b_shape = (output_size,)#ベクトルなので第二引数は未指定だが1になるはず
        b_initial_value = tf.zeros(b_shape)
        self.b = tf.Variable(b_initial_value)
        
    def __call__(self, inputs):#インスタンス名()で呼び出せる関数
        print("NaiveDense__call__ is called")#いつ呼ばれるのかチェック
        return self.activation(tf.matmul(inputs, self.W) + self.b)
    
    @property #関数を変数のように扱える. ()がないのでメンバにアクセスできなくなって安全
    def weights(self):
        return [self.W, self.b]

class NaiveSequential:
    def __init__(self, layers):
        self.layers = layers
        
    def __call__(self, inputs):
        x = inputs
        #xに入るのは[NaiveDense(inputsize,outputsize,relu),NaiveDense(,,softmax)]
        #つまりNaiveDense型の配列を渡して、xが__init__の引数に入る.
        for layer in self.layers:
            x = layer(x)
        return x
    @property
    def weights(self):
        weights = []
        for layer in self.layers:
            weights += layer.weights
        return weights#各Denseにおけるweightsを返す.

#ここからmain
model = NaiveSequential([
    NaiveDense(input_size=28 * 28, output_size = 512, activation=tf.nn.relu),
    NaiveDense(input_size=512, output_size = 10, activation=tf.nn.softmax)
])
assert len(model.weights) == 4

class BatchGenerator:
    def __init__(self, images, labels, batch_size=128):
        assert len(images) == len(labels)
        self.index = 0
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(images) / batch_size)
        
    def next(self):
        images = self.images[self.index : self.index + self.batch_size]
        labels = self.labels[self.index : self.index + self.batch_size]
        self.index += self.batch_size
        return images, labels

def one_training_step(model, images_batch, labels_batch):
    
    with tf.GradientTape() as tape:
        predictions = model(images_batch)
        per_sample_losses = tf.keras.losses.sparse_categorical_crossentropy(
            labels_batch, predictions)
        average_loss = tf.reduce_mean(per_sample_losses)
    gradients = tape.gradient(average_loss, model.weights)
    update_weights(gradients, model.weights)
    return average_loss

learning_rate = 1e-3

def update_weights(gradients, weights):
    for g, w in zip(gradients, weights):
        w.assign_sub(g * learning_rate)
        
optimizer = optimizers.SGD(learning_rate=1e-3)

def update_weights(gradients, weights):
    optimizer.apply_gradients(zip(gradients, weights))
    
def fit(model, images, labels, epochs, batch_size=128):
    for epoch_counter in range(epochs):
        print(f"Epoch {epoch_counter}")
        batch_generator = BatchGenerator(images, labels)
        for batch_counter in range(batch_generator.num_batches):
            images_batch, labels_batch = batch_generator.next()
            loss = one_training_step(model, images_batch, labels_batch)
            if batch_counter % 100 == 0:
                print(f"loss at batch {batch_counter}: {loss:.2f}")


#main
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

fit(model, train_images, train_labels, epochs=10, batch_size=128)

predictions = model(test_images)

predictions = predictions.numpy()
predicted_labels = np.argmax(predictions, axis=1)
matches = predicted_labels == test_labels
print(f"accuracy: {matches.mean():.2f}")