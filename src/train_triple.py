

"""

   CC BY-NC-ND 4.0 license
"""
import sys
from tools import *
from trainers import *
from datasets import *
import torchvision
import itertools
from common import *
import tensorboard
# from tensorboard import summary
from optparse import OptionParser

# for model parallel


parser = OptionParser()
parser.add_option('--gpu', type=int, help="gpu id", default=0)
parser.add_option('--resume', type=int, help="resume training?", default=0)
parser.add_option('--config', type=str, help="net configuration")
parser.add_option('--log', type=str, help="log path")

MAX_EPOCHS = 100000

def main(argv):
  (opts, args) = parser.parse_args(argv)

  # Load experiment setting
  assert isinstance(opts, object)
  config = NetConfig(opts.config)

  batch_size = config.hyperparameters['batch_size']
  max_iterations = config.hyperparameters['max_iterations']

  train_loader_a = get_data_loader(config.datasets['train_a'], batch_size) # for homo
  train_loader_b = get_data_loader(config.datasets['train_b'], batch_size) # for gt
  train_loader_c = get_data_loader(config.datasets['train_c'], batch_size) # for new

  trainer = []
  exec ("trainer=%s(config.hyperparameters)" % config.hyperparameters['trainer'])
  # trainer = COCOGANTrainer_triple(config.hyperparameters)
  # Check if resume training
  iterations = 0
  if opts.resume == 1:
    iterations = trainer.resume(config.snapshot_prefix)
  trainer.cuda(opts.gpu)
  if config.hyperparameters['para'] == 1:
    trainer.parallel()
  # trainer = nn.DataParallel(trainer, device_ids=range(4), output_device=3)


  ######################################################################################################################
  # Setup logger and repare image outputs
  # train_writer = tensorboard.FileWriter("%s/%s" % (opts.log,os.path.splitext(os.path.basename(opts.config))[0]))
  image_directory, snapshot_directory = prepare_snapshot_and_image_folder(config.snapshot_prefix, iterations, config.image_save_iterations)

  for ep in range(0, MAX_EPOCHS):
    for it, (images_a, images_b, images_c) in enumerate(itertools.izip(train_loader_a,train_loader_b, train_loader_c)):
      if images_a.size(0) != batch_size or images_b.size(0) != batch_size or images_c.size(0) != batch_size:
        continue
      images_a = Variable(images_a.cuda())
      images_b = Variable(images_b.cuda())
      images_c = Variable(images_c.cuda())

      # Main training code
      trainer.dis_update(images_a, images_b, images_c,  config.hyperparameters)
      image_outputs = trainer.gen_update(images_a, images_b, images_c, config.hyperparameters)
      assembled_images = trainer.assemble_outputs(images_a, images_b, image_outputs)

      # Dump training stats in log file
      if (iterations+1) % config.display == 0:
        write_loss(iterations, max_iterations, trainer)

      if (iterations+1) % config.image_save_iterations == 0:
        img_filename = '%s/gen_%08d.jpg' % (image_directory, iterations + 1)
        torchvision.utils.save_image(assembled_images.data / 2 + 0.5, img_filename, nrow=1)
        write_html(snapshot_directory + "/index.html", iterations + 1, config.image_save_iterations, image_directory)
      elif (iterations + 1) % config.image_display_iterations == 0:
        img_filename = '%s/gen.jpg' % (image_directory)
        torchvision.utils.save_image(assembled_images.data / 2 + 0.5, img_filename, nrow=1)

      # Save network weights
      if (iterations+1) % config.snapshot_save_iterations == 0:
        trainer.save(config.snapshot_prefix, iterations)

      iterations += 1
      if iterations >= max_iterations:
        return

if __name__ == '__main__':
  main(sys.argv)

