import tensorflow as tf
import tflearn
from keras.layers.convolutional import UpSampling3D
from tflearn.initializations import normal

from .utils import Network, ReLU, LeakyReLU


def convolve(opName, inputLayer, outputChannel, kernelSize, stride, stddev=1e-2, reuse=False,
             weights_init='uniform_scaling'):
    return tflearn.layers.conv_3d(inputLayer, outputChannel, kernelSize, strides=stride,
                                  padding='same', activation='linear', bias=True, scope=opName, reuse=reuse,
                                  weights_init=weights_init)


def convolveReLU(opName, inputLayer, outputChannel, kernelSize, stride, stddev=1e-2, reuse=False):
    return ReLU(convolve(opName, inputLayer,
                         outputChannel,
                         kernelSize, stride, stddev=stddev, reuse=reuse),
                opName + '_rectified')


def convolveLeakyReLU(opName, inputLayer, outputChannel, kernelSize, stride, alpha=0.1, stddev=1e-2, reuse=False):
    return LeakyReLU(convolve(opName, inputLayer,
                              outputChannel,
                              kernelSize, stride, stddev, reuse),
                     alpha, opName + '_leakilyrectified')


def upconvolve(opName, inputLayer, outputChannel, kernelSize, stride, targetShape, stddev=1e-2, reuse=False,
               weights_init='uniform_scaling'):
    return tflearn.layers.conv.conv_3d_transpose(inputLayer, outputChannel, kernelSize, targetShape, strides=stride,
                                                 padding='same', activation='linear', bias=False, scope=opName,
                                                 reuse=reuse, weights_init=weights_init)


def upconvolveReLU(opName, inputLayer, outputChannel, kernelSize, stride, targetShape, stddev=1e-2, reuse=False):
    return ReLU(upconvolve(opName, inputLayer,
                           outputChannel,
                           kernelSize, stride,
                           targetShape, stddev, reuse),
                opName + '_rectified')


def upconvolveLeakyReLU(opName, inputLayer, outputChannel, kernelSize, stride, targetShape, alpha=0.1, stddev=1e-2,
                        reuse=False):
    return LeakyReLU(upconvolve(opName, inputLayer,
                                outputChannel,
                                kernelSize, stride,
                                targetShape, stddev, reuse),
                     alpha, opName + '_rectified')


def _get_gaussian_kernel(sigma, filter_shape):
    """Compute 1D Gaussian kernel."""
    x = tf.range(-filter_shape // 2 + 1, filter_shape // 2 + 1)
    x = tf.cast(x ** 2, sigma.dtype)
    x = tf.exp(-x / (2.0 * (sigma ** 2)))
    x = x / tf.math.reduce_sum(x)
    return x


def gaussian_filter3d(sigma, filter_size):
    x = _get_gaussian_kernel(sigma, filter_size)
    y = _get_gaussian_kernel(sigma, filter_size)
    z = _get_gaussian_kernel(sigma, filter_size)

    xy = tf.repeat(tf.expand_dims(x, 0), repeats=[3], axis=0) * tf.reshape(y, [3, 1])
    xyz = tf.repeat(tf.expand_dims(xy, 0), repeats=[3], axis=0) * tf.reshape(z, [3, 1, 1])
    return tf.cast(xyz, tf.float32)


def separable_conv3d(input, filter):
    permuted_input = tf.transpose(input, perm=[4, 0, 1, 2, 3])

    outputs = tf.TensorArray(dtype=tf.float32, infer_shape=False, size=1, dynamic_size=True)
    i = tf.constant(0)

    condition = lambda i, input, outputs: tf.less(i, input.get_shape()[0])

    def body(i, input, outputs):
        # do something here which you want to do in your loop
        prepared = tf.transpose(permuted_input[i: i+1], perm=[1, 2, 3, 4, 0])
        applied = tf.nn.conv3d(prepared, filter, strides=[1, 1, 1, 1, 1], padding="SAME")
        # print('applied', applied.get_shape())
        outputs = outputs.write(i, tf.transpose(applied, perm=[4, 0, 1, 2, 3]))
        return [tf.add(i, 1), input, outputs]

    # do the loop:
    _, _, out = tf.while_loop(condition, body, [i, permuted_input, outputs])

    out = tf.transpose(out.concat(), perm=[1, 2, 3, 4, 0])
    out.set_shape((None, input.get_shape()[1], input.get_shape()[2], input.get_shape()[3], input.get_shape()[4]))
    print('out', out.get_shape())

    return out


def convolveLeakyReLUWithCurriculumSmoothing(opName, inputLayer, outputChannel, kernelSize, stride, sigma,
                                             alpha=0.1,
                                             stddev=1e-2, reuse=False):
    conv = convolve(opName, inputLayer,
                    outputChannel,
                    kernelSize, stride, stddev, reuse)
    print('conv', conv.get_shape())

    fil = tf.reshape(gaussian_filter3d(sigma, kernelSize), [kernelSize, kernelSize, kernelSize, 1, 1])

    conv_gaussian = separable_conv3d(conv, fil)
    print('gauss', conv_gaussian.get_shape())
    # conv_gaussian = conv

    return LeakyReLU(conv_gaussian, alpha, opName + '_leakilyrectified_gaussian')

def convolveLeakyReLU(opName, inputLayer, outputChannel, kernelSize, stride, alpha=0.1, stddev=1e-2, reuse=False):
    return LeakyReLU(convolve(opName, inputLayer,
                              outputChannel,
                              kernelSize, stride, stddev, reuse),
                     alpha, opName + '_leakilyrectified')


class VTN(Network):
    def __init__(self, name, flow_multiplier=1., channels=16, **kwargs):
        super().__init__(name, **kwargs)
        self.flow_multiplier = flow_multiplier
        self.channels = channels

    def build(self, img1, img2, gaussian_factor):
        '''
            img1, img2, flow : tensor of shape [batch, X, Y, Z, C]
        '''
        concatImgs = tf.concat([img1, img2], 4, 'concatImgs')

        dims = 3
        c = self.channels
        conv1 = convolveLeakyReLUWithCurriculumSmoothing(
            'conv1',   concatImgs, c,   3, 2, gaussian_factor)  # 64 * 64 * 64
        conv2 = convolveLeakyReLUWithCurriculumSmoothing(
            'conv2',   conv1,      c*2,   3, 2, gaussian_factor)  # 32 * 32 * 32
        conv3 = convolveLeakyReLUWithCurriculumSmoothing('conv3',   conv2,      c*4,   3, 2, gaussian_factor)
        conv3_1 = convolveLeakyReLUWithCurriculumSmoothing('conv3_1', conv3,      c*4,   3, 1, gaussian_factor)
        conv4 = convolveLeakyReLUWithCurriculumSmoothing(
            'conv4',   conv3_1,    c*8,  3, 2, gaussian_factor)  # 16 * 16 * 16
        conv4_1 = convolveLeakyReLUWithCurriculumSmoothing('conv4_1', conv4,      c*8,  3, 1, gaussian_factor)
        conv5 = convolveLeakyReLUWithCurriculumSmoothing(
            'conv5',   conv4_1,    c*16,  3, 2, gaussian_factor)  # 8 * 8 * 8
        conv5_1 = convolveLeakyReLUWithCurriculumSmoothing('conv5_1', conv5,      c*16,  3, 1, gaussian_factor)
        conv6 = convolveLeakyReLUWithCurriculumSmoothing(
            'conv6',   conv5_1,    c*32,  3, 2, gaussian_factor)  # 4 * 4 * 4
        conv6_1 = convolveLeakyReLUWithCurriculumSmoothing('conv6_1', conv6,      c*32,  3, 1, gaussian_factor)
        # 16 * 32 = 512 channels

        shape0 = concatImgs.shape.as_list()
        shape1 = conv1.shape.as_list()
        shape2 = conv2.shape.as_list()
        shape3 = conv3.shape.as_list()
        shape4 = conv4.shape.as_list()
        shape5 = conv5.shape.as_list()
        shape6 = conv6.shape.as_list()

        pred6 = convolve('pred6', conv6_1, dims, 3, 1)
        upsamp6to5 = upconvolve('upsamp6to5', pred6, dims, 4, 2, shape5[1:4])
        deconv5 = upconvolveLeakyReLU(
            'deconv5', conv6_1, shape5[4], 4, 2, shape5[1:4])
        concat5 = tf.concat([conv5_1, deconv5, upsamp6to5], 4, 'concat5')

        pred5 = convolve('pred5', concat5, dims, 3, 1)
        upsamp5to4 = upconvolve('upsamp5to4', pred5, dims, 4, 2, shape4[1:4])
        deconv4 = upconvolveLeakyReLU(
            'deconv4', concat5, shape4[4], 4, 2, shape4[1:4])
        concat4 = tf.concat([conv4_1, deconv4, upsamp5to4],
                            4, 'concat4')  # channel = 512+256+2

        pred4 = convolve('pred4', concat4, dims, 3, 1)
        upsamp4to3 = upconvolve('upsamp4to3', pred4, dims, 4, 2, shape3[1:4])
        deconv3 = upconvolveLeakyReLU(
            'deconv3', concat4, shape3[4], 4, 2, shape3[1:4])
        concat3 = tf.concat([conv3_1, deconv3, upsamp4to3],
                            4, 'concat3')  # channel = 256+128+2

        pred3 = convolve('pred3', concat3, dims, 3, 1)
        upsamp3to2 = upconvolve('upsamp3to2', pred3, dims, 4, 2, shape2[1:4])
        deconv2 = upconvolveLeakyReLU(
            'deconv2', concat3, shape2[4], 4, 2, shape2[1:4])
        concat2 = tf.concat([conv2, deconv2, upsamp3to2],
                            4, 'concat2')  # channel = 128+64+2

        pred2 = convolve('pred2', concat2, dims, 3, 1)
        upsamp2to1 = upconvolve('upsamp2to1', pred2, dims, 4, 2, shape1[1:4])
        deconv1 = upconvolveLeakyReLU(
            'deconv1', concat2, shape1[4], 4, 2, shape1[1:4])
        concat1 = tf.concat([conv1, deconv1, upsamp2to1], 4, 'concat1')
        pred0 = upconvolve('upsamp1to0', concat1, dims, 4, 2, shape0[1:4])

        return {'flow': pred0 * 20 * self.flow_multiplier}


class VoxelMorph(Network):
    def __init__(self, name, flow_multiplier=1., channels=16, **kwargs):
        super().__init__(name, **kwargs)
        self.flow_multiplier = flow_multiplier
        self.encoders = [m * channels for m in [1, 2, 2, 2]]
        self.decoders = [m * channels for m in [2, 2, 2, 2, 2, 1, 1]] + [3]

    def build(self, img1, img2, gaussian_factor):
        '''
            img1, img2, flow : tensor of shape [batch, X, Y, Z, C]
        '''
        concatImgs = tf.concat([img1, img2], 4, 'concatImgs')

        conv1 = convolveLeakyReLU(
            'conv1', concatImgs, self.encoders[0], 3, 2)  # 64 * 64 * 64
        conv2 = convolveLeakyReLU(
            'conv2', conv1, self.encoders[1], 3, 2)  # 32 * 32 * 32
        conv3 = convolveLeakyReLU(
            'conv3', conv2, self.encoders[2], 3, 2)  # 16 * 16 * 16
        conv4 = convolveLeakyReLU(
            'conv4', conv3, self.encoders[3], 3, 2)  # 8 * 8 * 8

        net = convolveLeakyReLU('decode4', conv4, self.decoders[0], 3, 1)
        net = tf.concat([UpSampling3D()(net), conv3], axis=-1)
        net = convolveLeakyReLU('decode3', net, self.decoders[1], 3, 1)
        net = tf.concat([UpSampling3D()(net), conv2], axis=-1)
        net = convolveLeakyReLU('decode2', net, self.decoders[2], 3, 1)
        net = tf.concat([UpSampling3D()(net), conv1], axis=-1)
        net = convolveLeakyReLU('decode1', net, self.decoders[3], 3, 1)
        net = convolveLeakyReLU('decode1_1', net, self.decoders[4], 3, 1)
        net = tf.concat([UpSampling3D()(net), concatImgs], axis=-1)
        net = convolveLeakyReLU('decode0', net, self.decoders[5], 3, 1)
        if len(self.decoders) == 8:
            net = convolveLeakyReLU('decode0_1', net, self.decoders[6], 3, 1)
        net = convolve(
            'flow', net, self.decoders[-1], 3, 1, weights_init=normal(stddev=1e-5))
        return {
            'flow': net * self.flow_multiplier
        }


def affine_flow(W, b, len1, len2, len3):
    b = tf.reshape(b, [-1, 1, 1, 1, 3])
    xr = tf.range(-(len1 - 1) / 2.0, len1 / 2.0, 1.0, tf.float32)
    xr = tf.reshape(xr, [1, -1, 1, 1, 1])
    yr = tf.range(-(len2 - 1) / 2.0, len2 / 2.0, 1.0, tf.float32)
    yr = tf.reshape(yr, [1, 1, -1, 1, 1])
    zr = tf.range(-(len3 - 1) / 2.0, len3 / 2.0, 1.0, tf.float32)
    zr = tf.reshape(zr, [1, 1, 1, -1, 1])
    wx = W[:, :, 0]
    wx = tf.reshape(wx, [-1, 1, 1, 1, 3])
    wy = W[:, :, 1]
    wy = tf.reshape(wy, [-1, 1, 1, 1, 3])
    wz = W[:, :, 2]
    wz = tf.reshape(wz, [-1, 1, 1, 1, 3])
    return (xr * wx + yr * wy) + (zr * wz + b)


def det3x3(M):
    M = [[M[:, i, j] for j in range(3)] for i in range(3)]
    return tf.add_n([
        M[0][0] * M[1][1] * M[2][2],
        M[0][1] * M[1][2] * M[2][0],
        M[0][2] * M[1][0] * M[2][1]
    ]) - tf.add_n([
        M[0][0] * M[1][2] * M[2][1],
        M[0][1] * M[1][0] * M[2][2],
        M[0][2] * M[1][1] * M[2][0]
    ])


class VTNAffineStem(Network):
    def __init__(self, name, flow_multiplier=1., **kwargs):
        super().__init__(name, **kwargs)
        self.flow_multiplier = flow_multiplier

    def build(self, img1, img2):
        '''
            img1, img2, flow : tensor of shape [batch, X, Y, Z, C]
        '''
        concatImgs = tf.concat([img1, img2], 4, 'coloncatImgs')

        dims = 3
        conv1 = convolveLeakyReLU(
            'conv1', concatImgs, 16, 3, 2)  # 64 * 64 * 64
        conv2 = convolveLeakyReLU(
            'conv2', conv1, 32, 3, 2)  # 32 * 32 * 32
        conv3 = convolveLeakyReLU('conv3', conv2, 64, 3, 2)
        conv3_1 = convolveLeakyReLU(
            'conv3_1', conv3, 64, 3, 1)
        conv4 = convolveLeakyReLU(
            'conv4', conv3_1, 128, 3, 2)  # 16 * 16 * 16
        conv4_1 = convolveLeakyReLU(
            'conv4_1', conv4, 128, 3, 1)
        conv5 = convolveLeakyReLU(
            'conv5', conv4_1, 256, 3, 2)  # 8 * 8 * 8
        conv5_1 = convolveLeakyReLU(
            'conv5_1', conv5, 256, 3, 1)
        conv6 = convolveLeakyReLU(
            'conv6', conv5_1, 512, 3, 2)  # 4 * 4 * 4
        conv6_1 = convolveLeakyReLU(
            'conv6_1', conv6, 512, 3, 1)
        ks = conv6_1.shape.as_list()[1:4]
        conv7_W = tflearn.layers.conv_3d(
            conv6_1, 9, ks, strides=1, padding='valid', activation='linear', bias=False, scope='conv7_W')
        conv7_b = tflearn.layers.conv_3d(
            conv6_1, 3, ks, strides=1, padding='valid', activation='linear', bias=False, scope='conv7_b')

        I = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
        W = tf.reshape(conv7_W, [-1, 3, 3]) * self.flow_multiplier
        b = tf.reshape(conv7_b, [-1, 3]) * self.flow_multiplier
        A = W + I
        # the flow is displacement(x) = place(x) - x = (Ax + b) - x
        # the model learns W = A - I.

        sx, sy, sz = img1.shape.as_list()[1:4]
        flow = affine_flow(W, b, sx, sy, sz)
        # determinant should be close to 1
        det = det3x3(A)
        det_loss = tf.nn.l2_loss(det - 1.0)
        # should be close to being orthogonal
        # C=A'A, a positive semi-definite matrix
        # should be close to I. For this, we require C
        # has eigen values close to 1 by minimizing
        # k1+1/k1+k2+1/k2+k3+1/k3.
        # to prevent NaN, minimize
        # k1+eps + (1+eps)^2/(k1+eps) + ...
        eps = 1e-5
        epsI = [[[eps * elem for elem in row] for row in Mat] for Mat in I]
        C = tf.matmul(A, A, True) + epsI

        def elem_sym_polys_of_eigen_values(M):
            M = [[M[:, i, j] for j in range(3)] for i in range(3)]
            sigma1 = tf.add_n([M[0][0], M[1][1], M[2][2]])
            sigma2 = tf.add_n([
                M[0][0] * M[1][1],
                M[1][1] * M[2][2],
                M[2][2] * M[0][0]
            ]) - tf.add_n([
                M[0][1] * M[1][0],
                M[1][2] * M[2][1],
                M[2][0] * M[0][2]
            ])
            sigma3 = tf.add_n([
                M[0][0] * M[1][1] * M[2][2],
                M[0][1] * M[1][2] * M[2][0],
                M[0][2] * M[1][0] * M[2][1]
            ]) - tf.add_n([
                M[0][0] * M[1][2] * M[2][1],
                M[0][1] * M[1][0] * M[2][2],
                M[0][2] * M[1][1] * M[2][0]
            ])
            return sigma1, sigma2, sigma3

        s1, s2, s3 = elem_sym_polys_of_eigen_values(C)
        ortho_loss = s1 + (1 + eps) * (1 + eps) * s2 / s3 - 3 * 2 * (1 + eps)
        ortho_loss = tf.reduce_sum(ortho_loss)

        return {
            'flow': flow,
            'W': W,
            'b': b,
            'det_loss': det_loss,
            'ortho_loss': ortho_loss
        }
