import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
from modules.model import FOMOnet

class LRP_FOMOnet(FOMOnet):

    def __init__(self, num_channels=4):
        super().__init__(num_channels)

    def lrp(self, x):
        # Forward pass
        out = self.forward(x)
        print(out)

        # Initialize relevance scores with the output
        relevance = out.clone()
        print(relevance)

        # Layer-wise relevance propagation
        relevance = self.lrp_backward(relevance, self.dconv1)
        relevance = self.lrp_backward(relevance, self.dconv2)
        relevance = self.lrp_backward(relevance, self.dconv3)
        relevance = self.lrp_backward(relevance, self.dconv4)
        relevance = self.lrp_backward(relevance, self.dconv5)
        relevance = self.lrp_backward(relevance, self.dconv6)

        # Return relevance scores
        return relevance

    def lrp_backward(self, relevance, layer):
        relevance = self.lrp_linear(layer, relevance)
        return relevance

    def lrp_linear(self, layer, relevance):
        layer_out = layer(relevance)
        layer_in = layer.out_channels

        if isinstance(layer, nn.Conv1d):
            layer_weights = layer.weight
            layer_bias = layer.bias
            layer_stride = layer.stride[0]
            layer_padding = layer.padding[0]

            relevance = self.lrp_conv1d(layer_out, relevance, layer_weights, layer_bias, layer_stride, layer_padding)
        elif isinstance(layer, nn.Linear):
            layer_weights = layer.weight
            layer_bias = layer.bias

            relevance = self.lrp_linear_layer(layer_out, relevance, layer_weights, layer_bias)

        return relevance

    def lrp_conv1d(self, layer_out, relevance, weights, bias, stride, padding):
        _, _, in_length = relevance.size()
        _, _, out_length = layer_out.size()

        relevance_padded = F.pad(relevance, (padding, padding))
        weights_flipped = torch.flip(weights, dims=[2])

        unfold_relevance = F.unfold(relevance_padded, (weights.size(2),), stride=stride)
        unfold_relevance = unfold_relevance.view(-1, in_length, weights.size(2))

        unfold_relevance *= weights_flipped.unsqueeze(0)
        unfold_relevance = unfold_relevance.sum(dim=2)

        relevance = F.fold(unfold_relevance, (out_length,), (1,), stride=stride)

        if bias is not None:
            relevance += bias.unsqueeze(0).unsqueeze(-1)

        return relevance

    def lrp_linear_layer(self, layer_out, relevance, weights, bias):
        relevance = relevance / (layer_out + 1e-9)  # Add epsilon to avoid division by zero

        relevance = relevance.matmul(weights)
        relevance = relevance.unsqueeze(-1)

        if bias is not None:
            relevance += bias.unsqueeze(0).unsqueeze(-1)

        return relevance