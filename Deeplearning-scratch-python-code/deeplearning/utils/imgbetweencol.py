
import numpy as np



def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):

  N, C, H, W = x_shape
  assert (H + 2 * padding - field_height) % stride == 0
  assert (W + 2 * padding - field_height) % stride == 0
  out_height = int((H + 2 * padding - field_height) / stride + 1)
  out_width = int((W + 2 * padding - field_width) / stride + 1)

  i0 = np.repeat(np.arange(field_height), field_width)
  i0 = np.tile(i0, C)
  i1 = stride * np.repeat(np.arange(out_height), out_width)
  j0 = np.tile(np.arange(field_width), field_height * C)
  j1 = stride * np.tile(np.arange(out_width), out_height)
  i = i0.reshape(-1, 1) + i1.reshape(1, -1)
  j = j0.reshape(-1, 1) + j1.reshape(1, -1)

  k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

  return (k, i, j)


def im2col_indices(x, field_height=3, field_width=3, padding=1, stride=1):
  """ An implementation of im2col based on some fancy indexing """
  p = padding
  x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

  k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                               stride)

  cols = x_padded[:, k, i, j]
  C = x.shape[1]
  cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
  return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):

  N, C, H, W = x_shape

  H_padded, W_padded = H + 2 * padding, W + 2 * padding
 

  x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)


  k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding,
                              stride=stride)


  cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)

  cols_reshaped = cols_reshaped.transpose(2, 0, 1)






  np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
  
  # print(cols_reshaped)

  operator_div=np.ones_like(x_padded)
  h=x_padded.shape[2]
  w=x_padded.shape[3]
  
  
  silce_dx=np.array(range(field_height-1,h-field_height+1,1)).reshape(1,-1)
  slice_dy=np.array(range(field_width-1,w-field_width+1,1)).reshape(1,-1)

  operator_div[:,:,silce_dx,:]=np.multiply(field_height,operator_div[:,:,silce_dx,:])
  operator_div[:,:,:,slice_dy]=np.multiply(field_width,operator_div[:,:,:,slice_dy])

  for i in range(field_height-1):
    operator_div[:,:,i,:]*=(i+1)

  for i in range(h-1,max(h-field_height,field_height-1),-1):
    operator_div[:,:,i,:]*=(h-i)

  for i in range(field_width-1):
    operator_div[:,:,:,i]*=(i+1)

  for i in range(w-1,max(w-field_width,field_width-1),-1):
    operator_div[:,:,:,i]*=(w-i)

  x_padded=np.divide(x_padded,operator_div)

  if padding == 0:
      return x_padded
  return x_padded[:, :, padding:-padding, padding:-padding]


# X=np.array(range(12)).reshape(1,1,3,4)
# cols=im2col_indices(X,field_height=3,field_width=3,padding=1,stride=1)
# Z=col2im_indices(cols,(1,1,3,4),field_height=3,field_width=3,padding=1,stride=1)
# print(Z)

def max_pool_forward_reshape(x, **kwagrs):

  N, C, H, W = x.shape
  pool_height, pool_width = kwagrs['pool_height'], kwagrs['pool_width']
  stride =kwagrs['stride']

  assert pool_height == pool_width == stride, 'Invalid pool params'
  assert H % pool_height == 0
  assert W % pool_height == 0

  x_reshaped = x.reshape(N, C, H // pool_height, pool_height,
                         W // pool_width, pool_width)

  out = x_reshaped.max(axis=3).max(axis=4)

  cache = (x, x_reshaped, out)
  return out, cache

def max_pool_backward_reshape(dout, cache):

  x, x_reshaped, out = cache

  dx_reshaped = np.zeros_like(x_reshaped)
  out_newaxis = out[:, :, :, np.newaxis, :, np.newaxis]
  mask = (x_reshaped == out_newaxis)
  dout_newaxis = dout[:, :, :, np.newaxis, :, np.newaxis]
  dout_broadcast, _ = np.broadcast_arrays(dout_newaxis, dx_reshaped)
  dx_reshaped[mask] = dout_broadcast[mask]
  dx_reshaped /= np.sum(mask, axis=(3, 5), keepdims=True)
  dx = dx_reshaped.reshape(x.shape)

  return dx