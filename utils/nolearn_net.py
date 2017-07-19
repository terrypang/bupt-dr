import numpy as np
import theano
import lasagne as nn
from lasagne.layers import get_output
from theano import tensor as T
from lasagne.layers import InputLayer
from nolearn.lasagne import NeuralNet as BaseNeuralNet


class NeuralNet(BaseNeuralNet):

    def transform(self, X, target_layer_name, transform=None, color_vec=None):
        target_layer = self.layers_[target_layer_name]

        layers = self.layers_
        input_layers = [
            layer for layer in layers.values()
            if isinstance(layer, nn.layers.InputLayer)
        ]
        X_inputs = [
            theano.In(input_layer.input_var, name=input_layer.name)
            for input_layer in input_layers
        ]

        target_layer_output = nn.layers.get_output(
            target_layer, None, deterministic=True
        )

        transform_iter = theano.function(
            inputs=X_inputs,
            outputs=target_layer_output,
            allow_input_downcast=True,
        )

        outputs = []
        for Xb, yb in self.batch_iterator_test(X, transform=transform, color_vec=color_vec):
            outputs.append(self.apply_batch_func(transform_iter, Xb))
        return np.vstack(outputs)
