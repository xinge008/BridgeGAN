"""

   CC BY-NC-ND 4.0 license
"""

"""
Args:
    inputs: (320,192) X 3
            a: homo
            b: gt
            c: new

    outputs:

"""

from common_net import *



class COCODis_triple_res(nn.Module):
    def __init__(self, params):
        super(COCODis_triple_res, self).__init__()
        ch = params['ch']
        input_dim_a = params['input_dim_a']
        input_dim_b = params['input_dim_b']
        input_dim_c = params['input_dim_c']
        input_dim_d = params['input_dim_c']

        n_layer = params['n_layer']
        self.model_A = self._make_net(ch, input_dim_a, n_layer)
        self.model_B = self._make_net(ch, input_dim_b, n_layer)     # for domain adaptation
        self.model_C = self._make_net(ch, input_dim_c, n_layer)
        self.model_D = self._make_net(ch, input_dim_c, n_layer)       #
        # self.model_coral = self._make_net(ch, input_dim_c, n_layer-3) # for domain adaptation

    def _make_net(self, ch, input_dim, n_layer):
        model = []
        model += [LeakyReLUConv2d(input_dim, ch, kernel_size=3, stride=2, padding=1)]  # 16
        tch = ch
        for i in range(1, n_layer):
            model += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)]  # 8
            tch *= 2
        model += [nn.Conv2d(tch, 1, kernel_size=1, stride=1, padding=0)]  # 1
        return nn.Sequential(*model)

    # def cuda(self, gpu):
    #     self.model_A.cuda(gpu)
    #     self.model_B.cuda(gpu)

    def forward(self, x_bA, x_aB, x_bC, x_cB):
        out_A = self.model_A(x_bA)
        # print "out_A:"
        # print out_A.size()
        out_A = out_A.view(-1)
        # print out_A.size()
        # print out_A
        outs_A = []
        outs_A.append(out_A)
        out_B = self.model_B(x_aB)
        out_B = out_B.view(-1)
        outs_B = []
        outs_B.append(out_B)

        out_C = self.model_C(x_bC)
        out_C = out_C.view(-1)
        outs_C = []
        outs_C.append(out_C)

        out_D = self.model_D(x_cB)
        out_D = out_D.view(-1)
        outs_D = []
        outs_D.append(out_D)

        return outs_A, outs_B, outs_C, outs_D

    # def forward_latent(self, x_source_target):
    #     # the input is composed of two parts: source and target concenated wrt 0
    #     # now the feature is extracted from the last convolutional layers and is reshaped
    #     out_1 = self.model_coral(x_source_target)
    #     out_1 = out_1.view(-1)
    #     # outs_1 = []
    #     # outs_1.append(out_1)
    #     return out_1

    def forward_domain(self, x_source_target):
        out_1 = self.model_B(x_source_target)
        out_1 = out_1.view(-1)
        # outs_1 = []
        # outs_1.append(out_1)
        return out_1


def run():
    import torch
    import torch.nn as nn
    from torch.autograd import Variable
    params = {}
    params['ch'] = 64
    params['input_dim_a'] = 3
    params['input_dim_b'] = 3
    params['input_dim_c'] = 3
    params['n_layer'] = 6
    params['n_enc_front_blk'] = 3

    params['n_enc_res_blk'] = 3

    params['n_enc_shared_blk'] = 1
    params['n_gen_shared_blk'] = 1
    params['n_gen_res_blk'] = 3

    params['n_gen_front_blk'] = 3

    COCOResGen_test = COCOResGen_triple_res(params).cuda()
    x_a = Variable(torch.FloatTensor(4, 3, 320, 192), requires_grad=False).cuda()
    x_b = Variable(torch.FloatTensor(4, 3, 320, 192), requires_grad=False).cuda()
    x_c = Variable(torch.FloatTensor(4, 3, 320, 192), requires_grad=False).cuda()
    a, b, c, d, e, f, h, g, w, shared = COCOResGen_test(x_a, x_b, x_c)
    print "x_C based tensor:"
    print h.size()
    # for the Dis

    COCODis_test = COCODis_triple_res(params).cuda()
    # print COCODis_test
    image_a, image_b, image_c = COCODis_test(x_a, x_b, x_c, x_c)
    print image_a[0].size()


class COCOResGen_triple_res(nn.Module):
    def __init__(self, params):
        super(COCOResGen_triple_res, self).__init__()
        input_dim_a = params['input_dim_a']
        input_dim_b = params['input_dim_b']

        input_dim_c = params['input_dim_c']

        ch = params['ch']
        n_enc_front_blk = params['n_enc_front_blk']
        n_enc_res_blk = params['n_enc_res_blk']
        n_enc_shared_blk = params['n_enc_shared_blk']
        n_gen_shared_blk = params['n_gen_shared_blk']
        n_gen_res_blk = params['n_gen_res_blk']
        n_gen_front_blk = params['n_gen_front_blk']
        encA = []
        encB = []

        encC = []
        # Encoders
        encA += [LeakyReLUConv2d(input_dim_a, ch, kernel_size=7, stride=1, padding=3)]
        encB += [LeakyReLUConv2d(input_dim_b, ch, kernel_size=7, stride=1, padding=3)]
        encC += [LeakyReLUConv2d(input_dim_c, ch, kernel_size=7, stride=1, padding=3)]
        tch = ch
        for i in range(1, n_enc_front_blk):
            encA += [ReLUINSConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)]
            encB += [ReLUINSConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)]
            encC += [ReLUINSConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)]
            tch *= 2
        for i in range(0, n_enc_res_blk):
            encA += [INSResBlock(tch, tch)]
            encB += [INSResBlock(tch, tch)]
            encC += [INSResBlock(tch, tch)]
        # Shared
        enc_shared = []
        for i in range(0, n_enc_shared_blk):
            enc_shared += [INSResBlock(tch, tch)]
        enc_shared += [GaussianNoiseLayer()]  # gaussianNoise is the (x + torch.randn(x.size()))
        dec_shared = []
        for i in range(0, n_gen_shared_blk):
            dec_shared += [INSResBlock(tch, tch)]
        # Decoders
        decA = []
        decB = []
        decC = []
        for i in range(0, n_gen_res_blk):
            decA += [INSResBlock(tch, tch)]
            decB += [INSResBlock(tch, tch)]
            decC += [INSResBlock(tch, tch)]
        for i in range(0, n_gen_front_blk - 1):
            decA += [ReLUINSConvTranspose2d(tch, tch // 2, kernel_size=3, stride=2, padding=1, output_padding=1)]
            decB += [ReLUINSConvTranspose2d(tch, tch // 2, kernel_size=3, stride=2, padding=1, output_padding=1)]
            decC += [ReLUINSConvTranspose2d(tch, tch // 2, kernel_size=3, stride=2, padding=1, output_padding=1)]
            tch = tch // 2
        decA += [nn.ConvTranspose2d(tch, input_dim_a, kernel_size=1, stride=1, padding=0)]
        decB += [nn.ConvTranspose2d(tch, input_dim_b, kernel_size=1, stride=1, padding=0)]
        decC += [nn.ConvTranspose2d(tch, input_dim_c, kernel_size=1, stride=1, padding=0)]
        decA += [nn.Tanh()]
        decB += [nn.Tanh()]
        decC += [nn.Tanh()]
        self.encode_A = nn.Sequential(*encA)
        self.encode_B = nn.Sequential(*encB)
        self.encode_C = nn.Sequential(*encC)
        self.enc_shared = nn.Sequential(*enc_shared)
        self.dec_shared = nn.Sequential(*dec_shared)
        self.decode_A = nn.Sequential(*decA)
        self.decode_B = nn.Sequential(*decB)
        self.decode_C = nn.Sequential(*decC)


    def forward(self, x_A, x_B, x_C):
        # encoder_XA = self.encode_A(x_A)
        # print encoder_XA.size()
        # print self.encode_A(x_A).size()
        # print self.encode_B(x_B).size()
        # print self.encode_C(x_C).size()
        out = torch.cat((self.encode_A(x_A), self.encode_B(x_B), self.encode_C(x_C)), 0)  #

        shared = self.enc_shared(out)
        # print "the output of shared encoder is "
        # print shared.size()
        out = self.dec_shared(shared)

        # print "the output of shared decoder is "
        # print out.size()

        # print out.size()
        out_A = self.decode_A(out)
        out_B = self.decode_B(out)
        out_C = self.decode_C(out)
        # print out_A.size()
        x_Aa, x_Ba, x_Ca = torch.split(out_A, x_A.size(0), dim=0)
        x_Ab, x_Bb, x_Cb = torch.split(out_B, x_A.size(0), dim=0)
        x_Ac, x_Bc, x_Cc = torch.split(out_C, x_A.size(0), dim=0)
        return x_Aa, x_Ba, x_Ca, x_Ab, x_Bb, x_Cb, x_Ac, x_Bc, x_Cc, shared

    def forward_a2b(self, x_A):
        out = self.encode_A(x_A)
        shared = self.enc_shared(out)
        out = self.dec_shared(shared)
        out = self.decode_B(out)
        return out, shared

    def forward_b2a(self, x_B):
        out = self.encode_B(x_B)
        shared = self.enc_shared(out)
        out = self.dec_shared(shared)
        out = self.decode_A(out)
        return out, shared

    def forward_c2b(self, x_C):
        out = self.encode_C(x_C)
        shared = self.enc_shared(out)
        out = self.dec_shared(shared)
        out = self.decode_B(out)
        return out, shared

    def forward_b2c(self, x_B):
        out = self.encode_B(x_B)
        shared = self.enc_shared(out)
        out = self.dec_shared(shared)
        out = self.decode_C(out)
        return out, shared

    def forward_coral(self, x_target):
        out = self.encode_A(x_target)
        shared = self.enc_shared(out)
        out = self.dec_shared(shared)
        out = self.decode_B(out)
        return out


if __name__ == '__main__':
    run()