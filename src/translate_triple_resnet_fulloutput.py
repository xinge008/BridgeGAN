#!/usr/bin/env python
"""
 
   CC BY-NC-ND 4.0 license   
"""
from common import *
import sys
import os
from trainers import *
import cv2
import torchvision
from tools import *
from optparse import OptionParser

parser = OptionParser()
parser.add_option('--trans_alone', type=int, help="showing the translated image alone", default=0)
parser.add_option('--a2b', type=int, help="1 for a2b and others for b2a", default=1)
parser.add_option('--gpu', type=int, help="gpu id", default=0)
parser.add_option('--config',type=str,help="net configuration")
parser.add_option('--weights',type=str,help="file location to the trained generator network weights")
parser.add_option('--output_folder',type=str,help="output image folder")


def write_img(output_data, output_image_name):
  output_img = output_data[0].data.cpu().numpy()

  if output_img.shape[0]!=1:
    output_img = np.expand_dims(output_img, 0)
  print output_img.shape
  new_output_img = np.transpose(output_img, [2, 3, 1, 0])
  new_output_img = new_output_img[:, :, :, 0]
  out_img = np.uint8(255 * (new_output_img / 2.0 + 0.5))
  out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
  cv2.imwrite(output_image_name, out_img)

def main(argv):
  (opts, args) = parser.parse_args(argv)

  # Load experiment setting
  assert isinstance(opts, object)
  config = NetConfig(opts.config)

  ######################################################################################################################
  # Read training parameters from the yaml file
  hyperparameters = {}
  for key in config.hyperparameters:
    print key
    exec ('hyperparameters[\'%s\'] = config.hyperparameters[\'%s\']' % (key, key))

  if opts.a2b==1:
    dataset = config.datasets['train_a']
    dataset2 = config.datasets['train_b']
    dataset3 = config.datasets['train_c']
  else:
    dataset = config.datasets['train_b']

  exec ("data = %s(dataset)" % dataset['class_name'])
  root = dataset['root']
  folder = dataset['folder']
  list = dataset['list_name']
  list_fullpath = os.path.join(root, list)
  with open(list_fullpath) as f:
    content = f.readlines()
  image_list = [x.strip().split(' ')[0] for x in content]
  image_list.sort()

  # exec ("data2 = %s(dataset)" % dataset2['class_name'])
  root2 = dataset2['root']
  folder2 = dataset2['folder']
  list2 = dataset2['list_name']
  list_fullpath2 = os.path.join(root2, list2)
  with open(list_fullpath2) as f2:
      content = f2.readlines()
  image_list2 = [x.strip().split(' ')[0] for x in content]
  image_list2.sort()

  # exec ("data3 = %s(dataset)" % dataset3['class_name'])
  root3 = dataset3['root']
  folder3 = dataset3['folder']
  list3 = dataset3['list_name']
  list_fullpath3 = os.path.join(root3, list3)
  with open(list_fullpath3) as f3:
      content = f3.readlines()
  image_list3 = [x.strip().split(' ')[0] for x in content]
  image_list3.sort()

  trainer = []
  exec ("trainer=%s(hyperparameters)" % hyperparameters['trainer'])

  # Prepare network
  trainer.gen.load_state_dict(torch.load(opts.weights))
  trainer.cuda(opts.gpu)
  # trainer.gen.eval()




  for image_name, image_name2, image_name3 in zip(image_list, image_list2, image_list3):
    print (image_name, image_name2, image_name3)
    full_img_name = os.path.join(root, folder, image_name)
    img = data._load_one_image(full_img_name,test=True)
    raw_data = img.transpose((2, 0, 1))  # convert to HWC
    final_data = torch.FloatTensor((raw_data / 255.0 - 0.5) * 2)
    final_data = final_data.contiguous()
    final_data = Variable(final_data.view(1,final_data.size(0),final_data.size(1),final_data.size(2))).cuda(opts.gpu)
    # trainer.gen.eval()

    full_img_name2 = os.path.join(root2, folder2, image_name2)
    img2 = data._load_one_image(full_img_name2, test=True)
    raw_data2 = img2.transpose((2, 0, 1))  # convert to HWC
    final_data2 = torch.FloatTensor((raw_data2 / 255.0 - 0.5) * 2)
    final_data2 = final_data2.contiguous()
    final_data2 = Variable(final_data2.view(1, final_data2.size(0), final_data2.size(1), final_data2.size(2))).cuda(opts.gpu)

    full_img_name3 = os.path.join(root3, folder3, image_name3)
    img3 = data._load_one_image(full_img_name3, test=True)
    raw_data3 = img3.transpose((2, 0, 1))  # convert to HWC
    final_data3 = torch.FloatTensor((raw_data3 / 255.0 - 0.5) * 2)
    final_data3 = final_data3.contiguous()
    final_data3 = Variable(final_data3.view(1, final_data3.size(0), final_data3.size(1), final_data3.size(2))).cuda(
        opts.gpu)

    if opts.a2b == 1:
      output_data = trainer.gen.forward_a2b(final_data)
      x_aa, x_ba, x_ca, x_ab, x_bb, x_cb, x_ac, x_bc, x_cc, shared = trainer.gen(final_data, final_data2, final_data3)

    else:
      output_data = trainer.gen.forward_b2a(final_data)

    output_image_name = os.path.join(opts.output_folder, image_name)
    directory = os.path.dirname(output_image_name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    if opts.trans_alone == 0:
      assembled_images = torch.cat((final_data, output_data[0]), 3)
      torchvision.utils.save_image(assembled_images.data / 2.0 + 0.5, output_image_name)
    else:
      # output_img = output_data[0].data.cpu().numpy()
      # new_output_img = np.transpose(output_img, [2, 3, 1, 0])
      # new_output_img = new_output_img[:, :, :, 0]
      # out_img = np.uint8(255 * (new_output_img / 2.0 + 0.5))
      # out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
      # cv2.imwrite(output_image_name, out_img)
      write_img(output_data, output_image_name)
      output_image_name_aa = os.path.join(opts.output_folder, image_name[0:-4] + "_aa.jpg")
      output_image_name_bb = os.path.join(opts.output_folder, image_name[0:-4] + "_bb.jpg")
      output_image_name_cc = os.path.join(opts.output_folder, image_name[0:-4] + "_cc.jpg")
      output_image_name_ab = os.path.join(opts.output_folder, image_name[0:-4] + "_ab.jpg")
      output_image_name_cb = os.path.join(opts.output_folder, image_name[0:-4] + "_cb.jpg")
      output_image_name_ba = os.path.join(opts.output_folder, image_name[0:-4] + "_ba.jpg")
      output_image_name_bc = os.path.join(opts.output_folder, image_name[0:-4] + "_bc.jpg")
      write_img(x_aa, output_image_name_aa)
      write_img(x_bb, output_image_name_bb)
      write_img(x_cc, output_image_name_cc)
      write_img(x_ab, output_image_name_ab)
      write_img(x_cb, output_image_name_cb)
      write_img(x_ba, output_image_name_ba)
      write_img(x_bc, output_image_name_bc)

  return 0


if __name__ == '__main__':
  main(sys.argv)

