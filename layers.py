import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim


def learned_upsample(x, batch_norm_params, mult=2, name='learned_upsample'):
    """
    Perform conv then upsample and conv again
    """
    
    sh = tf.shape(x)

    h = sh[1]
    w = sh[2]
    c = x.get_shape()[-1].value
    
    """
    x = slim.conv2d(x, c, [3,3],  
         padding='SAME',
         normalizer_fn=slim.batch_norm,
         normalizer_params=batch_norm_params,
         weights_initializer=tf.contrib.layers.xavier_initializer(False),
         activation_fn=lambda x: tf.nn.elu(x),
         scope=name+"_1"
    )
    x = tf.image.resize_images(x, [h*mult, w*mult], 
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    x = slim.conv2d(x, c, [1,1],
         padding='SAME',
         normalizer_fn=slim.batch_norm,
         normalizer_params=batch_norm_params,
         weights_initializer=tf.contrib.layers.xavier_initializer(False),
         activation_fn=lambda x: tf.nn.elu(x),
         scope=name+"_2"
    )
    """
    x = slim.conv2d(x, c * mult**2, [3,3],  
         padding='SAME',
         normalizer_fn=slim.batch_norm,
         normalizer_params=batch_norm_params,
         weights_initializer=tf.contrib.layers.xavier_initializer(False),
         activation_fn=lambda x: tf.nn.elu(x),
         scope=name+"_1"
    )


    x = tf.depth_to_space(x, mult, name=name)

    x = slim.conv2d(x, c, [1,1],
         padding='SAME',
         normalizer_fn=slim.batch_norm,
         normalizer_params=batch_norm_params,
         weights_initializer=tf.contrib.layers.xavier_initializer(False),
         activation_fn=lambda x: tf.nn.elu(x),
         scope=name+"_2"
    )
    
    return x

def learned_downsample(x, batch_norm_params, method="ps", name='learned_downsample'):
    """
    Downsample by factor of 2 with conv/pool
    """
    c = x.get_shape()[-1].value
    x = slim.conv2d(x, c, [3,3], scope=name + "_conv",
         padding='SAME',
         normalizer_fn=slim.batch_norm,
         normalizer_params=batch_norm_params,
         weights_initializer=tf.contrib.layers.xavier_initializer(False),
         activation_fn=lambda x: tf.nn.elu(x)
    )

    if method=="stride":
        x = tf.layers.max_pooling2d(x, 5, 2, padding='same', name=name+"_pool")
    elif method=="ps":
        x = tf.space_to_depth(x, 2, name=name+"_ps")
    else:
        raise ValueError('Unrecognized method: "%s"' % method)

    x = slim.conv2d(x, c, [1,1],
         padding='SAME',
         normalizer_fn=slim.batch_norm,
         normalizer_params=batch_norm_params,
         weights_initializer=tf.contrib.layers.xavier_initializer(False),
         activation_fn=lambda x: tf.nn.elu(x),
         scope=name+"_2"
    )
    return x

def res_block(inputs, batch_norm_params, name='res_block'):
    """
    resblock!
    """
    shortcut = inputs
    filters = inputs.get_shape()[-1]
    inputs = slim.conv2d(inputs, max(filters//2, 1), [1,1], scope=name+"_1",
         padding='SAME',
         normalizer_fn=slim.batch_norm,
         normalizer_params=batch_norm_params,
         weights_initializer=tf.contrib.layers.xavier_initializer(False),
         activation_fn=lambda x: tf.nn.elu(x)
    )
    inputs = slim.conv2d(inputs, filters, [3,3], scope=name+"_2",
         padding='SAME',
         normalizer_fn=slim.batch_norm,
         normalizer_params=batch_norm_params,
         weights_initializer=tf.contrib.layers.xavier_initializer(False),
         activation_fn=lambda x: tf.nn.elu(x)
    )
    inputs = tf.add(inputs, shortcut, name=name+"_add")
    return inputs


def wrap_pad_rows(x, n=1):
    """ Wrapping pad rows; zero pad columns"""
    out = tf.concat([x[:, -n:, :, :], x, x[:, :n, :, :]], axis=1)
    return tf.pad(out, [[0, 0], [0, 0], [1, 1], [0, 0]])


def polar_transformer(U, theta, out_size, name='polar_transformer',
                      radius_factor=0.7071, inverse=False):
    """Polar Transformer Layer

    Based on https://github.com/tensorflow/models/blob/master/transformer/spatial_transformer.py.
    _repeat(), _interpolate() are exactly the same;
    the polar transform implementation is in _transform()

    Args:
        U, theta, out_size, name: same as spatial_transformer.py
        log (bool): log-polar if True; else linear polar
        radius_factor (float): fraction of width that will be the maximum radius
    """
    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.transpose(
                tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
            rep = tf.cast(rep, 'int32')
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

    def _interpolate(im, x, y, out_size):
        with tf.variable_scope('_interpolate'):
            # constants
            num_batch = tf.shape(im)[0]
            height = tf.shape(im)[1]
            width = tf.shape(im)[2]
            channels = tf.shape(im)[3]

            x = tf.cast(x, 'float32')
            y = tf.cast(y, 'float32')
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            zero = tf.zeros([], dtype='int32')
            max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
            max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

            # scale indices from [-1, 1] to [0, width/height]
            x = (x + 1.0)*(width_f) / 2.0
            y = (y + 1.0)*(height_f) / 2.0

            # do sampling
            x0 = tf.cast(tf.floor(x), 'int32')
            x1 = x0 + 1
            y0 = tf.cast(tf.floor(y), 'int32')
            y1 = y0 + 1

            x0 = tf.clip_by_value(x0, zero, max_x)
            x1 = tf.clip_by_value(x1, zero, max_x)
            y0 = tf.clip_by_value(y0, zero, max_y)
            y1 = tf.clip_by_value(y1, zero, max_y)
            dim2 = width
            dim1 = width*height
            base = _repeat(tf.range(num_batch)*dim1, out_height*out_width)
            base_y0 = base + y0*dim2
            base_y1 = base + y1*dim2
            idx_a = base_y0 + x0
            idx_b = base_y1 + x0
            idx_c = base_y0 + x1
            idx_d = base_y1 + x1

            # use indices to lookup pixels in the flat image and restore
            # channels dim
            im_flat = tf.reshape(im, tf.stack([-1, channels]))
            im_flat = tf.cast(im_flat, 'float32')
            Ia = tf.gather(im_flat, idx_a)
            Ib = tf.gather(im_flat, idx_b)
            Ic = tf.gather(im_flat, idx_c)
            Id = tf.gather(im_flat, idx_d)

            # and finally calculate interpolated values
            x0_f = tf.cast(x0, 'float32')
            x1_f = tf.cast(x1, 'float32')
            y0_f = tf.cast(y0, 'float32')
            y1_f = tf.cast(y1, 'float32')
            wa = tf.expand_dims(((x1_f-x) * (y1_f-y)), 1)
            wb = tf.expand_dims(((x1_f-x) * (y-y0_f)), 1)
            wc = tf.expand_dims(((x-x0_f) * (y1_f-y)), 1)
            wd = tf.expand_dims(((x-x0_f) * (y-y0_f)), 1)
            output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
            return output

    def _meshgrid(height, width):
        with tf.variable_scope('_meshgrid'):
            x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                            tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
            y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                            tf.ones(shape=tf.stack([1, width])))

            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))
            grid = tf.concat(axis=0, values=[x_t_flat, y_t_flat])

            return grid

    def _transform(theta, input_dim, out_size):
        with tf.variable_scope('_transform'):
            shape = tf.shape(input_dim)
            num_batch = tf.shape(input_dim)[0]
            num_channels = input_dim.get_shape()[3]

            theta = tf.reshape(theta, (-1, 2))
            theta = tf.cast(theta, 'float32')

            out_height = out_size[0]
            out_width = out_size[1]
            grid = _meshgrid(out_height, out_width)
            grid = tf.expand_dims(grid, 0)
            grid = tf.reshape(grid, [-1])
            grid = tf.tile(grid, tf.stack([num_batch]))
            grid = tf.reshape(grid, tf.stack([num_batch, 2, -1]))
            
            W = tf.cast(tf.shape(input_dim)[2], 'float32')
            H = tf.cast(tf.shape(input_dim)[1], 'float32')
            maxR = W*radius_factor

            # if radius is from 1 to W/2; log R is from 0 to log(W/2)
            # we map the -1 to +1 grid to log R
            # then remap to 0 to 1
            
            # get radius in pix
            r_s_ = tf.exp((grid[:, 0, :] + 1)/2*tf.log(maxR))
            # convert it to [0, 2maxR/W]
            r_s = (r_s_ - 1) / (maxR - 1) * 2 * maxR / W

            # y is from -1 to 1; theta is from 0 to 2pi
            t_s = (grid[:, 1, :] + 1)*np.pi
            ## here we do the log-polar transform
            if not inverse:

                # use + theta[:, 0] to deal with origin
                x_s = r_s*tf.cos(t_s) + theta[:, 0, np.newaxis]
                y_s = r_s*tf.sin(t_s) + theta[:, 1, np.newaxis]
            else:

                # if radius is from 1 to W/2; log R is from 0 to log(W/2)
                # we map the -1 to +1 grid to log R
                # then remap to 0 to 1
                
                y_s = (tf.sin(t_s/np.pi) - theta[:, 1, np.newaxis]) / r_s
                x_s = (tf.cos(t_s/np.pi) - theta[:, 0, np.newaxis]) / r_s

            x_s_flat = tf.reshape(x_s, [-1])
            y_s_flat = tf.reshape(y_s, [-1])

            input_transformed = _interpolate(
                input_dim, x_s_flat, y_s_flat,
                out_size)
            output = tf.reshape(input_transformed, 
                    tf.stack([num_batch, out_height, out_width, num_channels]))
            return output
    with tf.variable_scope(name):
        output = _transform(theta, U, out_size)
        return output

def log(x, base):
    numerator = tf.log(x) 
    denominator = tf.log(base) 
    return numerator / denominator

def acot2(y, x):
    return np.pi/2 - tf.atan2(y, x)


if __name__ == '__main__':
    import cv2
    from matplotlib import pyplot as plt
    im = cv2.imread("dataset/CampusLoopDataset/live/Image001.jpg")/255.0
    h = im.shape[0]
    w = im.shape[1]
    center =  tf.placeholder_with_default(np.array([[0,0]],dtype=np.float32),
            shape=[1,2], name='center')
    x = tf.expand_dims(tf.placeholder_with_default(im.astype(np.float32), im.shape), 0)
    y = polar_transformer(x, center, [h,w])
    z = polar_transformer(y, center, [h,w], inverse=True)
    with tf.Session() as sess:
        p,im_ = sess.run([y,z])
    plt.subplot(3,1,1)
    plt.imshow(im)
    plt.subplot(3,1,2)
    plt.imshow(np.squeeze(p))
    plt.subplot(3,1,3)
    plt.imshow(np.squeeze(im_))
    plt.show()
