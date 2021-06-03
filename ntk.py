from jax import numpy as np
import numpy
import jax
import neural_tangents as nt
import tensorflow_datasets as tfds
import tensorflow as tf
from skimage.transform import resize
from math import pi, acos, sqrt
from copy import deepcopy
from PIL import Image

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

# normalize data to mean 0, norm 1 (ResNet paper normalizes data)
def normalize(data):
    img_sum = np.sum(data)
    # print(img_sum)
    data = data - img_sum/(32.*32.*3.)
    norm = np.linalg.norm(data)
    # print(norm)
    data = data/norm
    return data

def make_MNIST_shifted(data, n, img_size, corner = (0,0)):
  new_data = []
  for idx in range(n):
    new_img = numpy.zeros((2*img_size[0], 2*img_size[1], img_size[2]))
    new_img[corner[0]:corner[0]+img_size[0], corner[1]:corner[1]+img_size[1], :img_size[2]] = data[idx]
    new_data.append(new_img)
  return np.array(new_data)

dataset = 'cifar'

n_samples = 250 # training samples per class
n_test = 250 # test samples per class

# 2-class CIFAR dataset: 100 planes vs 100 ships
if dataset == 'cifar':
  # img_size = (8, 8, 3)
  img_size = (32, 32, 3)
  ds_train = tfds.as_numpy(tfds.load('cifar10', batch_size=-1, split='train'))
  x_train, y_train = ds_train['image'], ds_train['label']
  planes_x = x_train[y_train == 0]
  planes_y = y_train[y_train == 0]
  trucks_x = x_train[y_train == 9]
  trucks_y = y_train[y_train == 9]
  x_train = numpy.concatenate([planes_x[:n_samples], trucks_x[:n_samples]])
  x_train = numpy.array([normalize(resize(image, img_size)) for image in x_train])
  x_test = numpy.concatenate([planes_x[n_samples:n_samples+n_test], trucks_x[n_samples:n_samples+n_test]])
  x_test = numpy.array([normalize(resize(image, img_size)) for image in x_test])

# generate shifted MNIST dataset
elif dataset == 'mnist':
  # 2 class MNIST:
  ds_train = tfds.as_numpy(tfds.load('mnist', batch_size=-1, split='train'))
  x_train, y_train = ds_train['image'], ds_train['label']
  planes_x = x_train[y_train == 0]
  planes_y = y_train[y_train == 0]
  ships_x = x_train[y_train == 8]
  ships_y = y_train[y_train == 8]

  #img_size = (16, 16, 1)
  img_size = (8, 8, 1)

  x_train = numpy.concatenate([planes_x[:n_samples], ships_x[:n_samples]])
  x_train = numpy.array([resize(image, img_size) for image in x_train])
  x_train = make_MNIST_shifted(x_train, 2*n_samples, img_size)

  #Image.fromarray(numpy.squeeze((255*numpy.array(x_train[0])).astype(numpy.uint8),axis=2)).save('class1.jpg')
  #Image.fromarray(numpy.squeeze((255*numpy.array(x_train[-2])).astype(numpy.uint8), axis=2)).save('class2.jpg')

  x_test = numpy.concatenate([planes_x[n_samples:n_samples+n_test], ships_x[n_samples:n_samples+n_test]])
  x_test = numpy.array([resize(image, img_size) for image in x_test])
  x_test = make_MNIST_shifted(x_test, 2*n_test, img_size, (8,8))

  #Image.fromarray(numpy.squeeze((255*numpy.array(x_test[0])).astype(numpy.uint8),axis=2)).save('class1test.jpg')
  #Image.fromarray(numpy.squeeze((255*numpy.array(x_test[-2])).astype(numpy.uint8), axis=2)).save('class2test.jpg')

  img_size = (16, 16, 1)


# generate toy dataset:
elif dataset == 'toy':

  img_size = (8, 8, 3)

  x_train = numpy.zeros((2*n_samples, img_size[0], img_size[1], img_size[2]))
  for idx in range(n_samples):
    r = numpy.random.randint(0, 4)
    r=1
    c = numpy.random.randint(0, 4)
    c=1
    x_train[idx][r, c, 0] = 1.0
  for idx in range(n_samples, 2*n_samples):
    r = numpy.random.randint(0, 4)
    r=2
    c = numpy.random.randint(0, 4)
    c=0
    x_train[idx][r, c, 2] = 1.0
  x_train = np.array(x_train)

  x_test = numpy.zeros((2*n_test, img_size[0], img_size[1], img_size[2]))
  for idx in range(n_test):
    r = numpy.random.randint(4, 8)
    c = numpy.random.randint(4, 8)
    x_test[idx][r, c, 0] = 1.0
  for idx in range(n_test, 2*n_test):
    r = numpy.random.randint(4, 8)
    c = numpy.random.randint(4, 8)
    x_test[idx][r, c, 2] = 1.0
  x_test = np.array(x_test)

y_train = np.concatenate([np.ones((n_samples, 1)), -np.ones((n_samples, 1))])
y_test = np.concatenate([np.ones((n_test, 1)), -np.ones((n_test, 1))])

D = img_size[0]*img_size[1]*img_size[2]

# linear NTK:
def FCNLinear(depth, D, linear=True):
  # Neural_tangents automatically normalizes by 1/sqrt{d_i}. 
  # We need to undo this for the first layer to match other NTK definitions.
  layers = [nt.stax.Flatten(), nt.stax.Dense(1, D**(0.5))]
  if linear:
    w = 1.0
  else:
    w = 2**0.5
  for d in range(depth):
    if not linear:
      layers += [nt.stax.Relu()]
    layers += [nt.stax.Dense(1, w)]
  return nt.stax.serial(*layers)

def ConvNetLinear(depth, D, pool=False, linear=True, pad='SAME'):
  layers = []
  if linear:
    w = 1. # may need to change this for linear conv experiments at large depths
  else:
    w = 2**0.5
  for d in range(depth):
    layers += [nt.stax.Conv(1, (3, 3), strides=(1, 1), W_std=w, padding=pad)]
    if not linear:
      layers += [nt.stax.Relu()]
  if pool:
    layers += [nt.stax.GlobalAvgPool()]
  else:
    layers += [nt.stax.Flatten(), nt.stax.Dense(1, D**0.5)]
  return nt.stax.serial(*layers)

def train(kernel_fn, x_train, y_train, x_test, y_test):
  batched_kernel_fn = nt.batch(kernel_fn, 25)
  
  K_test_train = batched_kernel_fn(x_test, x_train).ntk
  
  K_train_train = batched_kernel_fn(x_train, x_train).ntk
  # NNGP = batched_kernel_fn(x_train, x_train).nngp
  # print(NNGP)
  #print(K_train_train)
  # print(K_train_train)
  y_test_pred = K_test_train @ np.linalg.inv(K_train_train) @ y_train
  #print(y_test_pred)
  loss_d = np.mean((y_test_pred - y_test)**2)
  y_test_class = np.where(y_test_pred > 0, 1., -1.)
  acc_d = np.mean(y_test_class == y_test)

  # y_train_pred = K_train_train @ np.linalg.inv(K_train_train) @ y_train
  # loss_t = np.mean((y_train_pred - y_train)**2)
  # y_train_class = np.where(y_train_pred > 0, 1., -1.)
  # acc_t = np.mean(y_train_class == y_train)

  # x_id = np.eye(D).reshape(D, img_size[0], img_size[1], img_size[2])
  # K_id_train = batched_kernel_fn(x_id, x_train).ntk
  # operator = K_id_train @ np.linalg.inv(K_train_train) @ y_train
  # norm = np.linalg.norm(operator)

  # batched_kernel_fn = nt.batch(kernel_fn, 32)

  # B_matrix = batched_kernel_fn(x_id, x_id).ntk
  # w, _ = numpy.linalg.eig(B_matrix)
  # condition_no = numpy.max(w)/numpy.min(w)

  return loss_d, acc_d
  # return loss_d, acc_d, loss_t, acc_t, norm, condition_no

#depths = [1, 5, 8, 10, 20, 50, 80, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
# depths = [1, 5, 8, 10]
depths = [1, 5, 10, 20, 30, 40, 50, 80, 100, 200, 500, 1000, 2000]
# depths = [500, 1000, 2000]
use_linear = True

# print("FCN")
# for depth in [1]:#depths:
#   _, _, FCN = FCNLinear(depth, D, linear=use_linear)
#   loss_d, acc_d, loss_t, acc_t, norm, kappa  = train(FCN, x_train, y_train, x_test)
#   print(f'depth: {depth}, acc: {acc_d}, loss: {loss_d}, train_acc: {acc_t}, train_loss: {loss_t}, norm: {norm}, {kappa}')

'''
print("CNN flat")
for depth in depths:
  _, _, CNN_flat = ConvNetLinear(depth, D, pool=False, linear=use_linear)
  loss_d, acc_d, loss_t, acc_t = train(CNN_flat, x_train, y_train, x_test)
  print(f'depth: {depth}, acc: {acc_d}, loss: {loss_d}, train_acc: {acc_t}, train_loss: {loss_t}')


print("CNN pool")
for depth in depths:
  _, _, CNN_pool = ConvNetLinear(depth, D, pool=True, linear=use_linear)
  # loss_d, acc_d, loss_t, acc_t, norm, condition_no = train(CNN_pool, x_train, y_train, x_test)
  # print(f'[{depth}, {acc_d}, {loss_d}, {acc_t}, {loss_t}, {norm}, {condition_no}],')
'''
print("NTK")
for depth in depths:
  _, _, FCN = FCNLinear(depth, D, linear=False)
  loss_d, acc_d = train(FCN, x_train, y_train, x_test, y_test)
  print(f'[{depth}, {acc_d}, {loss_d}],')

print("CNTK circular")
for depth in depths:
  _, _, CNN = ConvNetLinear(depth, D, pool=False, linear=False, pad='CIRCULAR')
  loss_d, acc_d = train(CNN, x_train, y_train, x_test, y_test)
  print(f'[{depth}, {acc_d}, {loss_d}],')

print("CNTK")
for depth in depths:
  _, _, CNN = ConvNetLinear(depth, D, pool=False, linear=False)
  loss_d, acc_d = train(CNN, x_train, y_train, x_test, y_test)
  print(f'[{depth}, {acc_d}, {loss_d}],')

# print("CNTK circular")
# for depth in depths:
#   _, _, CNN = ConvNetLinear(depth, D, pool=False, linear=False, pad='CIRCULAR')
#   loss_d, acc_d = train(CNN, x_train, y_train, x_test, y_test)
#   print(f'[{depth}, {acc_d}, {loss_d}],')
