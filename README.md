# Keras SVG Model plot library

This library was designed to help in the process of visualizing keras models graphs. It was expected to be as straightforward as possible.

## Prerequisites

* Keras
* Graphviz & Pydot

## HOW TO USE

First define your model using keras. Something like:

```
from keras.layers import Input,Conv2D,Dense,Flatten
from keras.models import Model

input = Input(shape=(100,100,1))
layer = Conv2D(64,(3,3),activation='relu')(input)
layer = Conv2D(128,(5,5),activation='relu')(layer)
layer = Flatten()(layer)
layer = Dense(128,activation='relu')(layer)
layer = Dense(32,activation='relu')(layer)
layer = Dense(1,activation='softmax')(layer)

model = Model(input,layer)
```

Now call the function to visualize the model (it does not require the model to be compiled nor trained, just created)
```
from K_model_plot import _get_model_Svg
_get_model_Svg(model,filename="my_model.svg", display_shapes=True, display_params=False)
```

This call will generate an svg image with a graph representing the model we just created. The "display_shapes" flag is used to toggle between displayinh or not the shape of the data through the layers of the net or not displaying them. In case the flag is set to True, the shape of the activations will be shown after any layer of the model that has the potential to effectively change the size of the data: convolutional, dense, pooling, flatten layers (activation, normalization, concatenate, merge, dropout layers are ignored). There is another flag that can be specified: "display_params", which is by default set to False. When this flag is set to True some important parameters of different layers of the model are displayed along with the layer itself (such as the kernel size and strides of a Conv2D layer, or the pool_size of a Pool layer, the dropout rate in a dropout layer, etc.). 

In the case of the example model previously defined the resulting SVG image would look like:

<p align="center">
 <img src="./my_model.png">
</p>

If the display_params flag was set to True in the previous example the result would look like:

<p align="center">
 <img src="./my_model_params.png">
</p>


## SUPPORTED LAYERS
Almost every keras layer is supported (unsupported layers are: SimpleRNNCell, GRUCell nor LSTMCell -which are usually wrapped inside an RNN, SimpleRNN, GRU, LSTM or ConvLSTM2D layer-. Layer wrappers (such as TimeDistributed or Bidirectional) are not supported either). See further documentation on Keras layers on https://keras.io/layers . The render for each type of layer is shown below:

### Core layers
From top to bottom and left to right: Input, Flatten, Dense, Lambda, ActivityRegularization, Masking, Reshape, Permute and RepeatVector layers.
<p align="center">
 <img src="./imgs/core_layers.png">
</p>

### Convolutional layers
<p align="center">
 <img src="./imgs/conv_layers.png">
</p>

### Pooling layers
<p align="center">
 <img src="./imgs/pool_layers.png">
</p>

### Locally Connected Layers
<p align="center">
 <img src="./imgs/locally_layers.png">
</p>

### Activation layers 
Notice that these layers are created using the Activation layer and specifying the activation function rather than using specific advanced activation layers. This means that the ReLU layer shown below was obtained using ```keras.layers.Activation('relu')``` instead of ```keras.layers.ReLU```.
<p align="center">
 <img src="./imgs/activation_layers.png">
</p>

### Advanced Activation Layers
<p align="center">
 <img src="./imgs/advance_activation_layers.png">
</p>

### Normalization Layers
<p align="center">
 <img src="./imgs/norm_layers.png">
</p>

### Dropout layers
<p align="center">
 <img src="./imgs/dropout_layers.png">
</p>

### Recurrent layers
<p align="center">
 <img src="./imgs/recurrent_layers.png">
</p>

### Noise layers
<p align="center">
 <img src="./imgs/noise_layers.png">
</p>

### Embedding layers
<p align="center">
 <img src="./imgs/embedding_layers.png">
</p>

### Merge layers
<p align="center">
 <img src="./imgs/merge_layers.png">
</p>

## CODE DEMO

A DEMO of this usage and the code can be found in jupyter notebook "Plot_Deep_Nets_DEMO.ipynb".

## Authors

* **Manuel Blanco Valentin** - http://github.com/manuelblancovalentin
* **Clecio R. Bom** - https://github.com/cdebom

 
