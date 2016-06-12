# Docker image for running Keras with ConvLSTM

Keras with ConvLSTM:
https://github.com/fchollet/keras/pull/1818

# Usage

On OSX, you will need to install [Docker Machine](https://docs.docker.com/machine/)
or boot2docker before running this image.

### Building the image:

```bash
docker build -t keras_convlstm .
docker run -p 8888:8888 -v $(pwd):/root/host keras_convlstm ipython notebook --ip=0.0.0.0 --no-browser
```

http://localhost:8888/


On OSX, change "localhost" to docker host ip:
<img width="575" alt="2016-06-11 18 07 39" src="https://cloud.githubusercontent.com/assets/1708549/15984255/a0e621f4-2fff-11e6-9af3-792128e24615.png">


### Sample:

```
import numpy as np
from keras.models import Sequential,Graph
from keras.layers.convolutional import Convolution2D,Convolution3D
from keras.layers.recurrent_convolutional import LSTMConv2D

seq = Sequential()
seq.add(LSTMConv2D(nb_filter=15, nb_row=3, nb_col=3, input_shape=(10,40,40,1),
                   border_mode="same",return_sequences=True))
seq.add(LSTMConv2D(nb_filter=15,nb_row=3, nb_col=3,
                   border_mode="same", return_sequences=True))
seq.add(LSTMConv2D(nb_filter=15, nb_row=3, nb_col=3,
                   border_mode="same", return_sequences=True))
seq.add(Convolution3D(nb_filter=1, kernel_dim1=1, kernel_dim2=3,
                      kernel_dim3=3, activation='sigmoid',
                   border_mode="same", dim_ordering="tf"))

seq.compile(loss="binary_crossentropy",optimizer="adadelta")

X_train = np.ones((320, 10,40,40,1))
Y_train = np.ones((320, 10,40,40,1))
seq.fit(X_train, Y_train, batch_size=32, verbose=1)
```
