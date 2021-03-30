import argparse
import os
import sys
import torch
sys.path.append('Wave-U-Net-Pytorch')
import data.utils
import model.utils as model_utils

from test import predict_song
from model.waveunet import Waveunet
from torch2trt import torch2trt, tensorrt_converter, add_missing_trt_tensors, trt, trt_version, get_arg


@tensorrt_converter('torch.nn.functional.conv_transpose1d', enabled=trt_version() >= '7.0')
@tensorrt_converter('torch.nn.functional.conv_transpose2d', enabled=trt_version() >= '7.0')
@tensorrt_converter('torch.nn.functional.conv_transpose3d', enabled=trt_version() >= '7.0')
def convert_Conv_trt7_functional(ctx):


    input = get_arg(ctx, 'input', pos=0, default=None)
    weight = get_arg(ctx, 'weight', pos=1, default=None)
    bias = get_arg(ctx, 'bias', pos=2, default=None)
    stride = get_arg(ctx, 'stride', pos=3, default=1)
    padding = get_arg(ctx, 'padding', pos=4, default=0)
    dilation = get_arg(ctx, 'dilation', pos=5, default=1)
    groups = get_arg(ctx, 'groups', pos=6, default=1)
    
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return

    input_dim = input.dim() - 2
    
    ndim = len(output.shape)

    out_channels = int(weight.shape[0])
    kernel_size = tuple(weight.shape[2:])
    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, ) * input_dim

    if not isinstance(stride, tuple):
        stride = (stride, ) * input_dim

    if not isinstance(padding, tuple):
        padding = (padding, ) * input_dim

    # if not isinstance(dilation, tuple):
    #     dilation = (dilation, ) * input_dim

    kernel = weight.detach().cpu().numpy()
    
    if bias is not None:
        bias = bias.detach().cpu().numpy()

    # reshape 1D to 2D to use 2D convolutions to accomplish 1D conv
    if input_dim == 1:
        layer = ctx.network.add_shuffle(input_trt)
        layer.reshape_dims = (-1, 1, input.shape[-1])
        input_trt = layer.get_output(0)
        kernel_size = (1,) + kernel_size
        stride = (1,) + stride
        padding = (0,) + padding
        # dilation = (1,) + dilation

    layer = ctx.network.add_deconvolution_nd(
        input=input_trt,
        num_output_maps=out_channels,
        kernel_shape=kernel_size,
        kernel=kernel,
        bias=bias)
    layer.stride_nd = stride
    layer.padding_nd = padding
    # layer.dilation_nd = dilation

    import pdb
    pdb.set_trace()
    if groups is not None:
        layer.num_groups = groups

    # reshape 1D from 2D back to 1D
    if input_dim == 1:
        layer = ctx.network.add_shuffle(layer.get_output(0))
        layer.reshape_dims = (-1, output.shape[-1])
        
    output._trt = layer.get_output(0)


@tensorrt_converter('torch.nn.functional.conv1d', enabled=trt_version() >= '7.0')
@tensorrt_converter('torch.nn.functional.conv2d', enabled=trt_version() >= '7.0')
@tensorrt_converter('torch.nn.functional.conv3d', enabled=trt_version() >= '7.0')
def convert_Conv_trt7_functional(ctx):

    input = get_arg(ctx, 'input', pos=0, default=None)
    weight = get_arg(ctx, 'weight', pos=1, default=None)
    bias = get_arg(ctx, 'bias', pos=2, default=None)
    stride = get_arg(ctx, 'stride', pos=3, default=1)
    padding = get_arg(ctx, 'padding', pos=4, default=0)
    dilation = get_arg(ctx, 'dilation', pos=5, default=1)
    groups = get_arg(ctx, 'groups', pos=6, default=1)
    
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return

    input_dim = input.dim() - 2
    
    ndim = len(output.shape)

    out_channels = int(weight.shape[0])
    kernel_size = tuple(weight.shape[2:])
    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, ) * input_dim

    if not isinstance(stride, tuple):
        stride = (stride, ) * input_dim

    if not isinstance(padding, tuple):
        padding = (padding, ) * input_dim

    if not isinstance(dilation, tuple):
        dilation = (dilation, ) * input_dim

    kernel = weight.detach().cpu().numpy()
    
    if bias is not None:
        bias = bias.detach().cpu().numpy()

    # reshape 1D to 2D to use 2D convolutions to accomplish 1D conv
    if input_dim == 1:
        layer = ctx.network.add_shuffle(input_trt)
        layer.reshape_dims = (-1, 1, input.shape[-1])
        input_trt = layer.get_output(0)
        kernel_size = (1,) + kernel_size
        stride = (1,) + stride
        padding = (0,) + padding
        dilation = (1,) + dilation

    layer = ctx.network.add_convolution_nd(
        input=input_trt,
        num_output_maps=out_channels,
        kernel_shape=kernel_size,
        kernel=kernel,
        bias=bias)
    layer.stride_nd = stride
    layer.padding_nd = padding
    layer.dilation_nd = dilation

    if groups is not None:
        layer.num_groups = groups

    # reshape 1D from 2D back to 1D
    if input_dim == 1:
        layer = ctx.network.add_shuffle(layer.get_output(0))
        layer.reshape_dims = (-1, output.shape[-1])
        
    output._trt = layer.get_output(0)


@tensorrt_converter('torch.nn.functional.pad')
def convert_pad(ctx):
    input = ctx.method_args[0]
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return
    
    pad = ctx.method_args[1]
    ndim = len(output.shape)

    # get tensorrt padding
    if len(pad) == 2:
        pre_padding = (0, pad[0])
        post_padding = (0, pad[1])
    elif len(pad) == 4:
        pre_padding = (pad[2], pad[0])
        post_padding = (pad[3], pad[1])
    else:
        raise NotImplementedError
    
    # reshape 1D to 2D
    if ndim == 3:
        layer = ctx.network.add_shuffle(input_trt)
        layer.reshape_dims = (-1, 1, input.shape[-1])
        input_trt = layer.get_output(0)

    layer = ctx.network.add_padding(input_trt, pre_padding, post_padding)

    # reshape 1D from 2D back to 1D
    if ndim == 3:
        layer = ctx.network.add_shuffle(layer.get_output(0))
        layer.reshape_dims = (-1, output.shape[-1])
    
    output._trt = layer.get_output(0)
    

def main(args):
    # MODEL
    num_features = [args.features*i for i in range(1, args.levels+1)] if args.feature_growth == "add" else \
                   [args.features*2**i for i in range(0, args.levels)]
    target_outputs = int(args.output_size * args.sr)
    model = Waveunet(args.channels, num_features, args.channels, args.instruments, kernel_size=args.kernel_size,
                     target_output_size=target_outputs, depth=args.depth, strides=args.strides,
                     conv_type=args.conv_type, res=args.res, separate=args.separate)

    print("Loading model from checkpoint " + str(args.load_model))
    state = model_utils.load_model(model, None, args.load_model, args.cuda)

    model = model.eval().cuda()

    data = torch.randn(1, 2, 97961).cuda()

    # convert submodules
    for key, module in model.waveunets_exec.items():
        print('Optimizing {key} with TensorRT...'.format(key=key))
        module_trt = torch2trt(module, [data], fp16_mode=True)
        print('Saving {key}...'.format(key=key))
        if not os.path.exists('trt_modules'):
            os.makedirs(model_trt)
        torch.save(module_trt.state_dict(), 'trt_modules/{key}_trt.pth'.format(key=key))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instruments', type=str, nargs='+', default=["bass", "drums", "other", "vocals"],
                        help="List of instruments to separate (default: \"bass drums other vocals\")")
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA (default: False)')
    parser.add_argument('--features', type=int, default=32,
                        help='Number of feature channels per layer')
    parser.add_argument('--load_model', type=str, default='checkpoints/waveunet/model',
                        help='Reload a previously trained model')
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size")
    parser.add_argument('--levels', type=int, default=6,
                        help="Number of DS/US blocks")
    parser.add_argument('--depth', type=int, default=1,
                        help="Number of convs per block")
    parser.add_argument('--sr', type=int, default=44100,
                        help="Sampling rate")
    parser.add_argument('--channels', type=int, default=2,
                        help="Number of input audio channels")
    parser.add_argument('--kernel_size', type=int, default=5,
                        help="Filter width of kernels. Has to be an odd number")
    parser.add_argument('--output_size', type=float, default=2.0,
                        help="Output duration")
    parser.add_argument('--strides', type=int, default=4,
                        help="Strides in Waveunet")
    parser.add_argument('--conv_type', type=str, default="gn",
                        help="Type of convolution (normal, BN-normalised, GN-normalised): normal/bn/gn")
    parser.add_argument('--res', type=str, default="fixed",
                        help="Resampling strategy: fixed sinc-based lowpass filtering or learned conv layer: fixed/learned")
    parser.add_argument('--separate', type=int, default=1,
                        help="Train separate model for each source (1) or only one (0)")
    parser.add_argument('--feature_growth', type=str, default="double",
                        help="How the features in each layer should grow, either (add) the initial number of features each time, or multiply by 2 (double)")

    parser.add_argument('--input', type=str, default=os.path.join("audio_examples", "Cristina Vane - So Easy", "mix.mp3"),
                        help="Path to input mixture to be separated")
    parser.add_argument('--output', type=str, default=None, help="Output path (same folder as input path if not set)")

    args = parser.parse_args()

    main(args)
