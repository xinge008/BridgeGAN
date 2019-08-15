"""
 
   CC BY-NC-ND 4.0 license   
"""
from cocogan_nets_triple import *
import numpy as np
from init import *
from helpers import get_model_list, _compute_fake_acc, _compute_true_acc
import torch
import torch.nn as nn
import os
import itertools


class COCOGANTrainer_triple(nn.Module):
  def __init__(self, hyperparameters):
    super(COCOGANTrainer_triple, self).__init__()
    lr = hyperparameters['lr']
    # Initiate the networks
    exec( 'self.dis = %s(hyperparameters[\'dis\'])' % hyperparameters['dis']['name'])
    exec( 'self.gen = %s(hyperparameters[\'gen\'])' % hyperparameters['gen']['name'] )
    # Setup the optimizers
    self.dis_opt = torch.optim.Adam(self.dis.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    # Network weight initialization
    self.dis.apply(gaussian_weights_init)
    self.gen.apply(gaussian_weights_init)
    # print self.gen
    # Setup the loss function for training
    self.ll_loss_criterion_a = torch.nn.L1Loss()
    self.ll_loss_criterion_b = torch.nn.L1Loss()
    self.ll_loss_criterion_c = torch.nn.L1Loss()


  def _compute_kl(self, mu):
    # def _compute_kl(self, mu, sd):
    # mu_2 = torch.pow(mu, 2)
    # sd_2 = torch.pow(sd, 2)
    # encoding_loss = (mu_2 + sd_2 - torch.log(sd_2)).sum() / mu_2.size(0)
    # return encoding_loss
    mu_2 = torch.pow(mu, 2)
    encoding_loss = torch.mean(mu_2)
    return encoding_loss

  def gen_update(self, images_a, images_b,images_c, hyperparameters):

    for para in self.dis.parameters():
        para.data.clamp_(-0.1, 0.1)


    self.gen.zero_grad()
    x_aa, x_ba, x_ca, x_ab, x_bb, x_cb, x_ac, x_bc, x_cc, shared = self.gen(images_a, images_b, images_c)
    if hyperparameters['para'] == 1:
      x_bab, shared_bab = self.gen.module.forward_a2b(x_ba)
      x_aba, shared_aba = self.gen.module.forward_b2a(x_ab)

      x_cbc, shared_cbc = self.gen.module.forward_b2c(x_cb)
      x_bcb, shared_bcb = self.gen.module.forward_c2b(x_bc)
    else:
      x_bab, shared_bab = self.gen.forward_a2b(x_ba)
      x_aba, shared_aba = self.gen.forward_b2a(x_ab)
      x_cbc, shared_cbc = self.gen.forward_b2c(x_cb)
      x_bcb, shared_bcb = self.gen.forward_c2b(x_bc)

    outs_a, outs_b, outs_c, outs_d = self.dis(x_ba,x_ab, x_bc, x_cb)
    for it, (out_a, out_b, out_c, out_d) in enumerate(itertools.izip(outs_a, outs_b, outs_c, outs_d)):
      outputs_a = nn.functional.sigmoid(out_a)
      outputs_b = nn.functional.sigmoid(out_b)

      outputs_c = nn.functional.sigmoid(out_c)
      outputs_d = nn.functional.sigmoid(out_d)

      all_ones = Variable(torch.from_numpy(np.random.uniform(1.0,1.0,size=(outputs_a.size())))).float().cuda()
      # all_ones = Variable(torch.ones((outputs_a.size(0))).cuda(self.gpu))
      if it==0:
        ad_loss_a = nn.functional.binary_cross_entropy(outputs_a, all_ones)
        ad_loss_b = nn.functional.binary_cross_entropy(outputs_b, all_ones)
        ad_loss_c = nn.functional.binary_cross_entropy(outputs_c, all_ones)
        ad_loss_d = nn.functional.binary_cross_entropy(outputs_d, all_ones)
      else:
        ad_loss_a += nn.functional.binary_cross_entropy(outputs_a, all_ones)
        ad_loss_b += nn.functional.binary_cross_entropy(outputs_b, all_ones)
        ad_loss_c += nn.functional.binary_cross_entropy(outputs_c, all_ones)
        ad_loss_d += nn.functional.binary_cross_entropy(outputs_d, all_ones)
    # gen has KL_loss and L1_loss

    # print shared

    enc_loss  = self._compute_kl(shared)
    enc_bab_loss = self._compute_kl(shared_bab)
    enc_aba_loss = self._compute_kl(shared_aba)
    enc_bcb_loss = self._compute_kl(shared_bcb)
    enc_cbc_loss = self._compute_kl(shared_cbc)


    ll_loss_a = self.ll_loss_criterion_a(x_aa, images_a)
    ll_loss_b = self.ll_loss_criterion_b(x_bb, images_b)
    ll_loss_c = self.ll_loss_criterion_c(x_cc, images_c)


    ll_loss_aba = self.ll_loss_criterion_a(x_aba, images_a)
    ll_loss_bab = self.ll_loss_criterion_b(x_bab, images_b)

    l1_loss_cbc = self.ll_loss_criterion_c(x_cbc, images_c)
    l1_loss_bcb = self.ll_loss_criterion_b(x_bcb, images_b)


    total_loss_1 = hyperparameters['gan_w'] * (ad_loss_a + ad_loss_b + ad_loss_c + ad_loss_d) + \
                 hyperparameters['ll_direct_link_w'] * (ll_loss_a + ll_loss_b + ll_loss_c) + \
                 hyperparameters['ll_cycle_link_w'] * (ll_loss_aba + ll_loss_bab + l1_loss_bcb + l1_loss_cbc) + \
                 hyperparameters['kl_direct_link_w'] * (enc_loss + enc_loss + enc_loss) + \
                 hyperparameters['kl_cycle_link_w'] * (enc_bab_loss + enc_aba_loss + enc_bcb_loss + enc_cbc_loss)
    # total_loss_1 has no L1_loss about the images_c; no l1_loss_c and l1_loss_cbc
    total_loss_2 = hyperparameters['gan_w'] * (ad_loss_a + ad_loss_b + ad_loss_c + ad_loss_d) + \
                 hyperparameters['ll_direct_link_w'] * (ll_loss_a + ll_loss_b) + \
                 hyperparameters['ll_cycle_link_w'] * (ll_loss_aba + ll_loss_bab + l1_loss_bcb ) + \
                 hyperparameters['kl_direct_link_w'] * (enc_loss + enc_loss + enc_loss) + \
                 hyperparameters['kl_cycle_link_w'] * (enc_bab_loss + enc_aba_loss + enc_bcb_loss + enc_cbc_loss)

    # total_loss_2 has no L1_loss about the images_c and images_cb; no l1_loss_bcb
    total_loss = hyperparameters['gan_w'] * (ad_loss_a + ad_loss_b) + \
                 0.1 *  (ad_loss_c + ad_loss_d) + \
                 hyperparameters['ll_direct_link_w'] * (ll_loss_a + ll_loss_b) + \
                 1 * (l1_loss_bcb + l1_loss_cbc + ll_loss_c) + \
                 hyperparameters['ll_cycle_link_w'] * (ll_loss_aba + ll_loss_bab ) + \
                 hyperparameters['kl_direct_link_w'] * (enc_loss + enc_loss + enc_loss) + \
                 hyperparameters['kl_cycle_link_w'] * (enc_bab_loss + enc_aba_loss ) + \
                 0.01 * ( enc_bcb_loss + enc_cbc_loss )

    total_loss.backward()




    print "total_loss: ", total_loss.data.cpu().numpy()[0]
    print "l1_loss_a: ", ll_loss_a.data.cpu().numpy()[0]
    print "l1_loss_b: ", ll_loss_b.data.cpu().numpy()[0]
    print "ll_loss_c: ", ll_loss_c.data.cpu().numpy()[0]
    print "ll_loss_ab: ", self.ll_loss_criterion_a(x_ab, images_b).data.cpu().numpy()[0]
    print "ll_loss_cb: ", self.ll_loss_criterion_a(x_cb, images_b).data.cpu().numpy()[0]
    # print "ll_loss_d: ", l1_loss
    self.gen_opt.step()
    self.gen_enc_loss = enc_loss.data.cpu().numpy()[0]
    self.gen_enc_bab_loss = enc_bab_loss.data.cpu().numpy()[0]
    self.gen_enc_aba_loss = enc_aba_loss.data.cpu().numpy()[0]
    self.gen_enc_cbc_loss = enc_cbc_loss.data.cpu().numpy()[0]
    self.gen_enc_bcb_loss = enc_bcb_loss.data.cpu().numpy()[0]

    self.gen_ad_loss_a = ad_loss_a.data.cpu().numpy()[0]
    self.gen_ad_loss_b = ad_loss_b.data.cpu().numpy()[0]
    self.gen_ad_loss_c = ad_loss_c.data.cpu().numpy()[0]

    self.gen_ll_loss_a = ll_loss_a.data.cpu().numpy()[0]
    self.gen_ll_loss_b = ll_loss_b.data.cpu().numpy()[0]
    self.gen_ll_loss_c = ll_loss_c.data.cpu().numpy()[0]


    self.gen_ll_loss_aba = ll_loss_aba.data.cpu().numpy()[0]
    self.gen_ll_loss_bab = ll_loss_bab.data.cpu().numpy()[0]
    self.gen_ll_loss_cbc = l1_loss_cbc.data.cpu().numpy()[0]
    self.gen_ll_loss_bcb = l1_loss_bcb.data.cpu().numpy()[0]

    """
        #test all loss:
    """
    # print "gen_enc_loss: ", self.gen_enc_loss
    # print "gen_enc_bab_loss: ", self.gen_enc_bab_loss
    # print self.gen_enc_aba_loss
    # print self.gen_enc_cbc_loss
    # print self.gen_enc_bcb_loss
    # print self.gen_ad_loss_a
    # print self.gen_ad_loss_b
    # print self.gen_ad_loss_c
    # print self.gen_ll_loss_a
    # print self.gen_ll_loss_b, self.gen_ll_loss_c
    # print self.gen_ll_loss_aba, self.gen_ll_loss_bab, self.gen_ll_loss_cbc, self.gen_ll_loss_bcb
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



    self.gen_total_loss = total_loss.data.cpu().numpy()[0]
    return (x_aa, x_ba, x_ca,  x_ab, x_bb, x_cb, x_aba, x_bab, x_bcb, x_cbc)

  def dis_update(self, images_a, images_b, images_c, hyperparameters):

    # weight clipping
    for para in self.dis.parameters():
        para.data.clamp_(-0.1, 0.1)

    self.dis.zero_grad()
    x_aa, x_ba, x_ca, x_ab, x_bb, x_cb, x_ac, x_bc, x_cc, shared = self.gen(images_a, images_b, images_c)
    data_a = torch.cat((images_a, x_ba), 0)
    data_b = torch.cat((images_b, x_ab), 0)
    data_c = torch.cat((images_c, x_bc), 0)
    data_d = torch.cat((images_b, x_cb), 0)
    res_a, res_b, res_c, res_d = self.dis(data_a,data_b, data_c, data_d)

    # res_true_a, res_true_b = self.dis(images_a,images_b)
    # res_fake_a, res_fake_b = self.dis(x_ba, x_ab)
    for it, (this_a, this_b, this_c, this_d) in enumerate(itertools.izip(res_a, res_b, res_c, res_d)):
      out_a = nn.functional.sigmoid(this_a)
      out_b = nn.functional.sigmoid(this_b)

      out_c = nn.functional.sigmoid(this_c)
      out_d = nn.functional.sigmoid(this_d)


      # print out_a.size()
      out_true_a, out_fake_a = torch.split(out_a, out_a.size(0) // 2, 0)
      out_true_b, out_fake_b = torch.split(out_b, out_b.size(0) // 2, 0)

      out_true_c, out_fake_c = torch.split(out_c, out_c.size(0) // 2, 0)
      out_true_d, out_fake_d = torch.split(out_d, out_d.size(0) // 2, 0)

      # print out_true_a.size()
      out_true_n = out_true_a.size(0)
      out_fake_n = out_fake_a.size(0)
      # all1 = Variable(torch.ones((out_true_n)).cuda(self.gpu))
      tmp_all1 = torch.from_numpy(np.random.uniform(1.0,1.0, size=out_true_n))
      # print tmp_all1
      # print out_true_a
      all1 = Variable(tmp_all1).float().cuda()
      # all0 = Variable(torch.zeros((out_fake_n)).cuda(self.gpu))
      tmp_all0 = torch.from_numpy(np.random.uniform(0.0, 0.0, size=out_fake_n))
      all0 = Variable(tmp_all0).float().cuda()
      # dis only contains classification loss
      ad_true_loss_a = nn.functional.binary_cross_entropy(out_true_a, all1)
      ad_true_loss_b = nn.functional.binary_cross_entropy(out_true_b, all1)
      ad_true_loss_c = nn.functional.binary_cross_entropy(out_true_c, all1)
      ad_true_loss_d = nn.functional.binary_cross_entropy(out_true_d, all1)

      ad_fake_loss_a = nn.functional.binary_cross_entropy(out_fake_a, all0)
      ad_fake_loss_b = nn.functional.binary_cross_entropy(out_fake_b, all0)
      ad_fake_loss_c = nn.functional.binary_cross_entropy(out_fake_c, all0)
      ad_fake_loss_d = nn.functional.binary_cross_entropy(out_fake_d, all0)




      if it==0:
        ad_loss_a = ad_true_loss_a + ad_fake_loss_a
        ad_loss_b = ad_true_loss_b + ad_fake_loss_b
        ad_loss_c = ad_true_loss_c + ad_fake_loss_c
        ad_loss_d = ad_true_loss_d + ad_fake_loss_d
      else:
        ad_loss_a += ad_true_loss_a + ad_fake_loss_a
        ad_loss_b += ad_true_loss_b + ad_fake_loss_b
        ad_loss_c += ad_true_loss_c + ad_fake_loss_c
        ad_loss_d += ad_true_loss_d + ad_fake_loss_d


      true_a_acc = _compute_true_acc(out_true_a)
      true_b_acc = _compute_true_acc(out_true_b)
      true_c_acc = _compute_true_acc(out_true_c)
      true_d_acc = _compute_true_acc(out_true_d)

      fake_a_acc = _compute_fake_acc(out_fake_a)
      fake_b_acc = _compute_fake_acc(out_fake_b)
      fake_c_acc = _compute_fake_acc(out_fake_c)
      fake_d_acc = _compute_fake_acc(out_fake_d)
      exec( 'self.dis_true_acc_a2b_%d = 0.5 * (true_a_acc + true_b_acc)' %it)
      exec( 'self.dis_fake_acc_a2b_%d = 0.5 * (fake_a_acc + fake_b_acc)' %it)
      exec ('self.dis_true_acc_c2b_%d = 0.5 * (true_c_acc + true_d_acc)' % it)
      exec ('self.dis_fake_acc_b2c_%d = 0.5 * (fake_c_acc + fake_d_acc)' % it)

    loss = hyperparameters['gan_w'] * ( ad_loss_a + ad_loss_b + ad_loss_c + ad_loss_d )
    loss.backward()
    self.dis_opt.step()
    self.dis_loss = loss.data.cpu().numpy()[0]
    print "dis_loss: ", self.dis_loss
    print "dis_true_acc(a2b): ", 0.5 * (true_a_acc + true_b_acc)
    print "dis_fake_acc(a2b): ", 0.5 * (fake_a_acc + fake_b_acc)
    print "dis_true_acc(c2b): ", 0.5 * (true_c_acc + true_d_acc)
    print "dis_fake_acc(c2b): ", 0.5 * (fake_c_acc + fake_d_acc)
    return

  def assemble_outputs(self, images_a, images_b, network_outputs):
    images_a = self.normalize_image(images_a)
    images_b = self.normalize_image(images_b)
    x_aa = self.normalize_image(network_outputs[0])
    x_ba = self.normalize_image(network_outputs[1])
    x_ab = self.normalize_image(network_outputs[2])
    x_bb = self.normalize_image(network_outputs[3])
    x_aba = self.normalize_image(network_outputs[4])
    x_bab = self.normalize_image(network_outputs[5])
    return torch.cat((images_a[0:1, ::], x_aa[0:1, ::], x_ab[0:1, ::], x_aba[0:1, ::],
                      images_b[0:1, ::], x_bb[0:1, ::], x_ba[0:1, ::], x_bab[0:1, ::]), 3)

  def resume(self, snapshot_prefix):
    dirname = os.path.dirname(snapshot_prefix)
    last_model_name = get_model_list(dirname,"gen")
    if last_model_name is None:
      return 0
    self.gen.load_state_dict(torch.load(last_model_name))
    last0 = last_model_name.rfind("0")
    iterations = int(last_model_name[last0-7:last0+1])
    # iterations = int(last_model_name[5:13])
    last_model_name = get_model_list(dirname, "dis")
    self.dis.load_state_dict(torch.load(last_model_name))
    print('Resume from iteration %d' % iterations)
    return iterations

  def save(self, snapshot_prefix, iterations):
    gen_filename = '%s_gen_%08d_mini_clip.pkl' % (snapshot_prefix, iterations + 1)
    dis_filename = '%s_dis_%08d_mini_clip.pkl' % (snapshot_prefix, iterations + 1)
    torch.save(self.gen.state_dict(), gen_filename)
    torch.save(self.dis.state_dict(), dis_filename)

  def cuda(self, gpu):
    self.gpu = gpu
    self.dis.cuda()
    self.gen.cuda()

  def parallel(self):
    self.dis = nn.DataParallel(self.dis, device_ids=range(2), output_device=0)
    self.gen = nn.DataParallel(self.gen, device_ids=range(2), output_device=0)
    # self.dis_opt = nn.DataParallel(self.dis_opt, device_ids=range(4), output_device=0)
    # self.gen_opt = nn.DataParallel(self.gen_opt, device_ids=range(4), output_device=0)


  def normalize_image(self, x):
    return x[:,0:3,:,:]
