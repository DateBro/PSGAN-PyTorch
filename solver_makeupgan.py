import datetime
import os
import os.path as osp
import time

import torch.nn.init as init
import torchvision.models as models
from torchvision.utils import save_image

import makeup_gan
import tools.plot as plot_fig
from data_loaders.makeup_utils import *
from ops.histogram_matching import *
from ops.loss_added import GANLoss

pwd = osp.split(osp.realpath(__file__))[0]


class Solver_MakeupGAN(object):
    def __init__(self, data_loaders, config, dataset_config):
        self.checkpoint = config.checkpoint
        # Hyper-parameteres
        self.g_lr = config.G_LR
        self.d_lr = config.D_LR
        self.ndis = config.ndis
        self.num_epochs = config.num_epochs  # set 50
        self.num_epochs_decay = config.num_epochs_decay
        self.batch_size = config.batch_size
        self.norm = config.norm

        # Training settings
        self.snapshot_step = config.snapshot_step
        self.log_step = config.log_step
        self.vis_step = config.vis_step
        self.task_name = config.task_name

        # Data loader
        self.data_loader_train = data_loaders[0]
        self.data_loader_test = data_loaders[1]

        # Model hyper-parameters
        self.img_size = config.img_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.d_repeat_num = config.d_repeat_num
        self.lips = config.lips
        self.skin = config.skin
        self.eye = config.eye

        # Hyper-parameteres
        self.lambda_idt = config.lambda_idt
        self.lambda_A = config.lambda_A
        self.lambda_B = config.lambda_B
        self.lambda_his_lip = config.lambda_his_lip
        self.lambda_his_skin_1 = config.lambda_his_skin_1
        self.lambda_his_skin_2 = config.lambda_his_skin_2
        self.lambda_his_eye = config.lambda_his_eye
        self.lambda_vgg = config.lambda_vgg

        self.beta1 = config.beta1
        self.beta2 = config.beta2

        self.cls = config.cls_list
        self.content_layer = config.content_layer
        self.direct = config.direct
        # Test settings
        self.test_model = config.test_model

        # Path
        self.log_path = config.log_path + '_' + config.task_name
        self.vis_path = config.vis_path + '_' + config.task_name
        self.snapshot_path = config.snapshot_path + '_' + config.task_name
        self.result_path = config.vis_path + '_' + config.task_name

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        if not os.path.exists(self.vis_path):
            os.makedirs(self.vis_path)
        if not os.path.exists(self.snapshot_path):
            os.makedirs(self.snapshot_path)

        self.build_model()
        # Start with trained model
        if self.checkpoint:
            self.load_checkpoint()

        # for recording
        self.start_time = time.time()
        # epoch和iteration
        self.e = 0
        self.i = 0
        self.loss = {}

        # dataloader
        # The number of iterations per epoch
        self.iters_per_epoch = len(self.data_loader_train)

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        if not os.path.exists(self.vis_path):
            os.makedirs(self.vis_path)
        if not os.path.exists(self.snapshot_path):
            os.makedirs(self.snapshot_path)

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def update_lr(self, g_lr, d_lr):
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for i in self.cls:
            for param_group in getattr(self, "d_" + i + "_optimizer").param_groups:
                param_group['lr'] = d_lr

    def log_terminal(self):
        elapsed = time.time() - self.start_time
        elapsed = str(datetime.timedelta(seconds=elapsed))

        log = "Elapsed [{}], Epoch [{}/{}], Iter [{}/{}]".format(
            elapsed, self.e + 1, self.num_epochs, self.i + 1, self.iters_per_epoch)

        for tag, value in self.loss.items():
            log += ", {}: {:.4f}".format(tag, value)
        print(log)

    def save_models(self):
        torch.save(self.G.state_dict(),
                   os.path.join(self.snapshot_path, '{}_{}_G.pth'.format(self.e + 1, self.i + 1)))
        for i in self.cls:
            torch.save(getattr(self, "D_" + i).state_dict(),
                       os.path.join(self.snapshot_path, '{}_{}_D_'.format(self.e + 1, self.i + 1) + i + '.pth'))

    def weights_init_xavier(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.xavier_normal(m.weight.data, gain=1.0)
        elif classname.find('Linear') != -1:
            init.xavier_normal(m.weight.data, gain=1.0)

    def to_var(self, x, requires_grad=True):
        if torch.cuda.is_available():
            x = x.cuda()
        if not requires_grad:
            return Variable(x, requires_grad=requires_grad)
        else:
            return Variable(x)

    def de_norm(self, x):
        out = (x + 1) / 2
        return out.clamp(0, 1)

    def load_checkpoint(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.snapshot_path, '{}_G.pth'.format(self.checkpoint))))
        for i in self.cls:
            getattr(self, "D_" + i).load_state_dict(torch.load(os.path.join(
                self.snapshot_path, '{}_D_'.format(self.checkpoint) + i + '.pth')))
        print('loaded trained models (step: {})..!'.format(self.checkpoint))

    def build_model(self):
        # Define generators and discriminators
        self.G = makeup_gan.Generator()
        for i in self.cls:
            setattr(self, "D_" + i,
                    makeup_gan.Discriminator(self.img_size, self.d_conv_dim, self.d_repeat_num, self.norm))

        self.criterionL1 = torch.nn.L1Loss()
        self.criterionL2 = torch.nn.MSELoss()
        self.criterionGAN = GANLoss(use_lsgan=True, tensor=torch.cuda.FloatTensor)
        self.vgg = models.vgg16(pretrained=True)
        # Optimizers
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        for i in self.cls:
            setattr(self, "d_" + i + "_optimizer",
                    torch.optim.Adam(filter(lambda p: p.requires_grad, getattr(self, "D_" + i).parameters()), \
                                     self.d_lr, [self.beta1, self.beta2]))

        # Weights initialization
        self.G.apply(self.weights_init_xavier)
        for i in self.cls:
            getattr(self, "D_" + i).apply(self.weights_init_xavier)

        # Print networks
        self.print_network(self.G, 'G')
        for i in self.cls:
            self.print_network(getattr(self, "D_" + i), "D_" + i)

        if torch.cuda.is_available():
            self.G.cuda()
            self.vgg.cuda()
            for i in self.cls:
                getattr(self, "D_" + i).cuda()

    def vgg_forward(self, model, x):
        for i in range(18):
            x = model.features[i](x)
        return x

    def rebound_box(self, mask_A, mask_B, mask_A_face):
        index_tmp = mask_A.nonzero()
        x_A_index = index_tmp[:, 2]
        y_A_index = index_tmp[:, 3]
        index_tmp = mask_B.nonzero()
        x_B_index = index_tmp[:, 2]
        y_B_index = index_tmp[:, 3]
        mask_A_temp = mask_A.copy_(mask_A)
        mask_B_temp = mask_B.copy_(mask_B)
        mask_A_temp[:, :, min(x_A_index) - 10:max(x_A_index) + 11, min(y_A_index) - 10:max(y_A_index) + 11] = \
            mask_A_face[:, :, min(x_A_index) - 10:max(x_A_index) + 11, min(y_A_index) - 10:max(y_A_index) + 11]
        mask_B_temp[:, :, min(x_B_index) - 10:max(x_B_index) + 11, min(y_B_index) - 10:max(y_B_index) + 11] = \
            mask_A_face[:, :, min(x_B_index) - 10:max(x_B_index) + 11, min(y_B_index) - 10:max(y_B_index) + 11]
        mask_A_temp = self.to_var(mask_A_temp, requires_grad=False)
        mask_B_temp = self.to_var(mask_B_temp, requires_grad=False)
        return mask_A_temp, mask_B_temp

    def mask_preprocess(self, mask_A, mask_B):
        index_tmp = mask_A.nonzero()
        x_A_index = index_tmp[:, 2]
        y_A_index = index_tmp[:, 3]
        index_tmp = mask_B.nonzero()
        x_B_index = index_tmp[:, 2]
        y_B_index = index_tmp[:, 3]
        mask_A = self.to_var(mask_A, requires_grad=False)
        mask_B = self.to_var(mask_B, requires_grad=False)
        index = [x_A_index, y_A_index, x_B_index, y_B_index]
        index_2 = [x_B_index, y_B_index, x_A_index, y_A_index]
        return mask_A, mask_B, index, index_2

    def criterionHis(self, input_data, target_data, mask_src, mask_tar, index):
        input_data = (self.de_norm(input_data) * 255).squeeze()
        target_data = (self.de_norm(target_data) * 255).squeeze()
        mask_src = mask_src.expand(1, 3, mask_src.size(2), mask_src.size(2)).squeeze()
        mask_tar = mask_tar.expand(1, 3, mask_tar.size(2), mask_tar.size(2)).squeeze()
        input_masked = input_data * mask_src
        target_masked = target_data * mask_tar
        input_match = histogram_matching(input_masked, target_masked, index)
        input_match = self.to_var(input_match, requires_grad=False)
        loss = self.criterionL1(input_masked, input_match)
        return loss

    @staticmethod
    def generate(org_A, ref_B, mask_A=None, mask_B=None, gamma=None, beta=None, ret=False, generator=None,
                 mode='train'):
        """org_A is content, ref_B is style"""
        G = generator
        res = G(org_A, ref_B, mask_A, mask_B, gamma, beta, ret, mode)
        return res

    def train(self):
        # Start with trained model if exists
        cls_A = self.cls[0]
        cls_B = self.cls[1]
        g_lr = self.g_lr
        d_lr = self.d_lr
        if self.checkpoint:
            start = int(self.checkpoint.split('_')[0])
            self.vis_test()
        else:
            start = 0
        # Start training
        self.start_time = time.time()
        for self.e in range(start, self.num_epochs):
            current_iter = 0
            for self.i, (img_A, img_B, mask_A, mask_B) in enumerate(self.data_loader_train):
                # Convert tensor to variable
                # mask attribute: 0:background 1:face 2:left-eyebrown 3:right-eyebrown 4:left-eye 5: right-eye 6: nose
                # 7: upper-lip 8: teeth 9: under-lip 10:hair 11: left-ear 12: right-ear 13: neck
                if self.checkpoint or self.direct:
                    if self.lips:
                        mask_A_lip = (mask_A == 7).float() + (mask_A == 9).float()
                        mask_B_lip = (mask_B == 7).float() + (mask_B == 9).float()
                        mask_A_lip, mask_B_lip, index_A_lip, index_B_lip = self.mask_preprocess(mask_A_lip, mask_B_lip)
                    if self.skin:
                        mask_A_skin = (mask_A == 1).float() + (mask_A == 6).float() + (mask_A == 13).float()
                        mask_B_skin = (mask_B == 1).float() + (mask_B == 6).float() + (mask_B == 13).float()
                        mask_A_skin, mask_B_skin, index_A_skin, index_B_skin = self.mask_preprocess(mask_A_skin,
                                                                                                    mask_B_skin)
                    if self.eye:
                        mask_A_eye_left = (mask_A == 4).float()
                        mask_A_eye_right = (mask_A == 5).float()
                        mask_B_eye_left = (mask_B == 4).float()
                        mask_B_eye_right = (mask_B == 5).float()
                        mask_A_face = (mask_A == 1).float() + (mask_A == 6).float()
                        mask_B_face = (mask_B == 1).float() + (mask_B == 6).float()
                        # avoid the situation that images with eye closed
                        if not ((mask_A_eye_left > 0).any() and (mask_B_eye_left > 0).any() and
                                (mask_A_eye_right > 0).any() and (mask_B_eye_right > 0).any()):
                            continue
                        mask_A_eye_left, mask_A_eye_right = self.rebound_box(mask_A_eye_left, mask_A_eye_right,
                                                                             mask_A_face)
                        mask_B_eye_left, mask_B_eye_right = self.rebound_box(mask_B_eye_left, mask_B_eye_right,
                                                                             mask_B_face)
                        # 这里可以修改成同时计算双眼的loss，因为有时双眼的妆是看起来不对称的
                        mask_A_eye = mask_A_eye_left + mask_A_eye_right
                        mask_B_eye = mask_B_eye_left + mask_B_eye_right
                        mask_A_eye, mask_B_eye, index_A_eye, index_B_eye = self.mask_preprocess(mask_A_eye, mask_B_eye)
                try:
                    processed_img_A = Image.open(img_A[0])
                    processed_img_B = Image.open(img_B[0])
                    processed_img_A = preprocess_makeup_gan(processed_img_A)
                    processed_img_B = preprocess_makeup_gan(processed_img_B)
                    processed_org_A = [self.to_var(item, requires_grad=False) for item in processed_img_A]
                    processed_ref_B = [self.to_var(item, requires_grad=False) for item in processed_img_B]
                except Exception as e:
                    print(str(e))
                    print('current iteration is: ', current_iter)
                    print('image_A is: ', img_A[0])
                    print('image_B is: ', img_B[0])
                    continue

                org_A = processed_org_A[0]
                ref_B = processed_ref_B[0]

                # ================== Train D ================== #
                # training D_A, D_A aims to distinguish class B
                # Real
                out = getattr(self, "D_" + cls_A)(ref_B)
                d_loss_real = self.criterionGAN(out, True)
                # Fake
                fake_A = Solver_MakeupGAN.generate(processed_org_A[0], processed_ref_B[0], processed_org_A[1],
                                                   processed_ref_B[1], generator=self.G)
                fake_B = Solver_MakeupGAN.generate(processed_ref_B[0], processed_org_A[0], processed_ref_B[1],
                                                   processed_org_A[1], generator=self.G)
                fake_A = Variable(fake_A.data).detach()
                fake_B = Variable(fake_B.data).detach()
                out = getattr(self, "D_" + cls_A)(fake_A)
                d_loss_fake = self.criterionGAN(out, False)

                # Backward + Optimize
                d_loss = (d_loss_real + d_loss_fake) * 0.5
                getattr(self, "d_" + cls_A + "_optimizer").zero_grad()
                d_loss.backward(retain_graph=True)
                getattr(self, "d_" + cls_A + "_optimizer").step()

                # Logging
                self.loss = {'D-A-loss_real': d_loss_real.item()}

                # training D_B, D_B aims to distinguish class A
                # Real
                out = getattr(self, "D_" + cls_B)(org_A)
                d_loss_real = self.criterionGAN(out, True)
                # Fake
                out = getattr(self, "D_" + cls_B)(fake_B)
                d_loss_fake = self.criterionGAN(out, False)

                # Backward + Optimize
                d_loss = (d_loss_real + d_loss_fake) * 0.5
                getattr(self, "d_" + cls_B + "_optimizer").zero_grad()
                d_loss.backward(retain_graph=True)
                getattr(self, "d_" + cls_B + "_optimizer").step()

                # Logging
                self.loss['D-B-loss_real'] = d_loss_real.item()

                # ================== Train G ================== #
                if (self.i + 1) % self.ndis == 0:
                    # adversarial loss, i.e. L_trans,v in the paper
                    # identity loss
                    # 论文里没有这个identity loss啊？
                    if self.lambda_idt > 0:
                        # G should be identity if ref_B or org_A is fed
                        idt_A1 = Solver_MakeupGAN.generate(processed_org_A[0], processed_org_A[0], processed_org_A[1],
                                                           processed_org_A[1], generator=self.G)
                        idt_A2 = Solver_MakeupGAN.generate(processed_org_A[0], processed_org_A[0], processed_org_A[1],
                                                           processed_org_A[1], generator=self.G)
                        idt_B1 = Solver_MakeupGAN.generate(processed_ref_B[0], processed_ref_B[0], processed_ref_B[1],
                                                           processed_ref_B[1], generator=self.G)
                        idt_B2 = Solver_MakeupGAN.generate(processed_ref_B[0], processed_ref_B[0], processed_ref_B[1],
                                                           processed_ref_B[1], generator=self.G)
                        # lambda_A和B都是啥？
                        loss_idt_A1 = self.criterionL1(idt_A1, org_A) * self.lambda_A * self.lambda_idt
                        loss_idt_A2 = self.criterionL1(idt_A2, org_A) * self.lambda_A * self.lambda_idt
                        loss_idt_B1 = self.criterionL1(idt_B1, ref_B) * self.lambda_B * self.lambda_idt
                        loss_idt_B2 = self.criterionL1(idt_B2, ref_B) * self.lambda_B * self.lambda_idt
                        # loss_idt
                        loss_idt = (loss_idt_A1 + loss_idt_A2 + loss_idt_B1 + loss_idt_B2) * 0.5
                    else:
                        loss_idt = 0

                    # GAN loss D_A(G_A(A))
                    # fake_A in class B
                    fake_A = Solver_MakeupGAN.generate(processed_org_A[0], processed_ref_B[0], processed_org_A[1],
                                                       processed_ref_B[1], generator=self.G)
                    fake_B = Solver_MakeupGAN.generate(processed_ref_B[0], processed_org_A[0], processed_ref_B[1],
                                                       processed_org_A[1], generator=self.G)

                    pred_fake = getattr(self, "D_" + cls_A)(fake_A)
                    g_A_loss_adv = self.criterionGAN(pred_fake, True)
                    # GAN loss D_B(G_B(B))
                    pred_fake = getattr(self, "D_" + cls_B)(fake_B)
                    g_B_loss_adv = self.criterionGAN(pred_fake, True)

                    rec_A = Solver_MakeupGAN.generate(fake_A, processed_org_A[0], processed_org_A[1],
                                                      processed_org_A[1], generator=self.G)
                    rec_B = Solver_MakeupGAN.generate(fake_B, processed_ref_B[0], processed_ref_B[1],
                                                      processed_ref_B[1], generator=self.G)

                    # color_histogram loss
                    # 这里作者的实现是不是有点问题啊，论文里的loss是计算的G(x,y)和HM(x,y)的，
                    # 也就是fake_A和HM(org_A, ref_B)的啊
                    # github issue 中作者也说这里和论文中little different
                    g_A_loss_his = 0
                    g_B_loss_his = 0
                    # 这里可以进行修改，这样是将左右眼分开计算loss，但如果是侧脸时
                    # 双眼的光影和妆的浓淡看起来是不对称的，可以改成同时计算双眼的makeup loss
                    if self.checkpoint or self.direct:
                        if self.lips:
                            g_A_lip_loss_his = self.criterionHis(fake_A, ref_B, mask_A_lip, mask_B_lip,
                                                                 index_A_lip) * self.lambda_his_lip
                            g_B_lip_loss_his = self.criterionHis(fake_B, org_A, mask_B_lip, mask_A_lip,
                                                                 index_B_lip) * self.lambda_his_lip
                            g_A_loss_his += g_A_lip_loss_his
                            g_B_loss_his += g_B_lip_loss_his
                        if self.skin:
                            g_A_skin_loss_his = self.criterionHis(fake_A, ref_B, mask_A_skin, mask_B_skin,
                                                                  index_A_skin) * self.lambda_his_skin_1
                            g_B_skin_loss_his = self.criterionHis(fake_B, org_A, mask_B_skin, mask_A_skin,
                                                                  index_B_skin) * self.lambda_his_skin_2
                            g_A_loss_his += g_A_skin_loss_his
                            g_B_loss_his += g_B_skin_loss_his
                        if self.eye:
                            g_A_eye_loss_his = self.criterionHis(fake_A, ref_B, mask_A_eye, mask_B_eye,
                                                                 index_A_eye) * self.lambda_his_eye
                            g_B_eye_loss_his = self.criterionHis(fake_B, org_A, mask_B_eye, mask_A_eye,
                                                                 index_B_eye) * self.lambda_his_eye
                            g_A_loss_his += g_A_eye_loss_his
                            g_B_loss_his += g_B_eye_loss_his

                        # cycle loss
                    g_loss_rec_A = self.criterionL1(rec_A, org_A) * self.lambda_A
                    g_loss_rec_B = self.criterionL1(rec_B, ref_B) * self.lambda_B

                    # vgg loss
                    vgg_org = self.vgg_forward(self.vgg, org_A)
                    vgg_org = Variable(vgg_org.data).detach()
                    vgg_fake_A = self.vgg_forward(self.vgg, fake_A)
                    g_loss_A_vgg = self.criterionL2(vgg_fake_A, vgg_org) * self.lambda_A * self.lambda_vgg

                    vgg_ref = self.vgg_forward(self.vgg, ref_B)
                    vgg_ref = Variable(vgg_ref.data).detach()
                    vgg_fake_B = self.vgg_forward(self.vgg, fake_B)
                    g_loss_B_vgg = self.criterionL2(vgg_fake_B, vgg_ref) * self.lambda_B * self.lambda_vgg

                    loss_rec = (g_loss_rec_A + g_loss_rec_B + g_loss_A_vgg + g_loss_B_vgg) * 0.5

                    # Combined loss
                    g_loss = g_A_loss_adv + g_B_loss_adv + loss_rec + loss_idt
                    if self.checkpoint or self.direct:
                        g_loss = g_A_loss_adv + g_B_loss_adv + loss_rec + loss_idt + g_A_loss_his + g_B_loss_his

                    self.g_optimizer.zero_grad()
                    g_loss.backward(retain_graph=True)
                    self.g_optimizer.step()

                    # Logging
                    self.loss['G-A-loss-adv'] = g_A_loss_adv.item()
                    self.loss['G-B-loss-adv'] = g_A_loss_adv.item()
                    self.loss['G-loss-org'] = g_loss_rec_A.item()
                    self.loss['G-loss-ref'] = g_loss_rec_B.item()
                    # self.loss['G-loss-idt'] = loss_idt.item()
                    self.loss['G-loss-idt'] = loss_idt
                    self.loss['G-loss-img-rec'] = (g_loss_rec_A + g_loss_rec_B).item()
                    self.loss['G-loss-vgg-rec'] = (g_loss_A_vgg + g_loss_B_vgg).item()
                    if self.direct:
                        self.loss['G-A-loss-his'] = g_A_loss_his.item()
                        self.loss['G-B-loss-his'] = g_B_loss_his.item()

                # Print out log info
                if (current_iter + 1) % self.log_step == 0:
                    self.log_terminal()

                # plot the figures
                for key_now in self.loss.keys():
                    plot_fig.plot(key_now, self.loss[key_now])

                # save the images
                if (current_iter + 1) % self.vis_step == 0:
                    print("Saving middle output...")
                    self.vis_train([org_A, ref_B, fake_A, fake_B, rec_A, rec_B])

                # Save model checkpoints
                if (current_iter + 1) % self.snapshot_step == 0:
                    self.save_models()
                if current_iter % 100 == 99:
                    plot_fig.flush(self.task_name)

                plot_fig.tick()

                current_iter += 1

            # Decay learning rate
            if (self.e + 1) > (self.num_epochs - self.num_epochs_decay):
                g_lr -= (self.g_lr / float(self.num_epochs_decay))
                d_lr -= (self.d_lr / float(self.num_epochs_decay))
                self.update_lr(g_lr, d_lr)
                print('Decay learning rate to g_lr: {}, d_lr:{}.'.format(g_lr, d_lr))

            if self.e % 20 == 0:
                print("Saving output...")
                self.vis_test()

    def vis_train(self, img_train_list):
        # saving training results
        mode = "train_vis"
        img_train_list = torch.cat(img_train_list, dim=3)
        result_path_train = os.path.join(self.result_path, mode)
        if not os.path.exists(result_path_train):
            os.mkdir(result_path_train)
        save_path = os.path.join(result_path_train, '{}_{}_fake.jpg'.format(self.e, self.i))
        save_image(self.de_norm(img_train_list.data), save_path, normalize=True)

    def vis_test(self):
        # saving test results
        mode = "test_vis"
        for i, (img_A, img_B) in enumerate(self.data_loader_test):
            if i == 20:
                print('vis_test 20 images finish')
                break
            try:
                processed_img_A = Image.open(img_A[0])
                processed_img_B = Image.open(img_B[0])
                processed_img_A = preprocess_makeup_gan(processed_img_A)
                processed_img_B = preprocess_makeup_gan(processed_img_B)
                processed_org_A = [self.to_var(item, requires_grad=False) for item in processed_img_A]
                processed_ref_B = [self.to_var(item, requires_grad=False) for item in processed_img_B]
            except Exception as e:
                print(str(e))
                print('vis_test image_A is: ', img_A[0])
                print('vis_test image_B is: ', img_B[0])
                continue

            real_org = processed_org_A[0]
            real_ref = processed_ref_B[0]

            image_list = [real_org, real_ref]

            # Get makeup result
            fake_A = Solver_MakeupGAN.generate(processed_org_A[0], processed_ref_B[0], processed_org_A[1],
                                               processed_ref_B[1], generator=self.G)
            fake_B = Solver_MakeupGAN.generate(processed_ref_B[0], processed_org_A[0], processed_ref_B[1],
                                               processed_org_A[1], generator=self.G)

            rec_A = Solver_MakeupGAN.generate(fake_A, processed_org_A[0], processed_org_A[1],
                                              processed_org_A[1], generator=self.G)
            rec_B = Solver_MakeupGAN.generate(fake_B, processed_ref_B[0], processed_ref_B[1],
                                              processed_ref_B[1], generator=self.G)

            image_list.append(fake_A)
            image_list.append(fake_B)
            image_list.append(rec_A)
            image_list.append(rec_B)

            image_list = torch.cat(image_list, dim=3)
            vis_train_path = os.path.join(self.result_path, mode)
            result_path_now = os.path.join(vis_train_path, "epoch" + str(self.e))
            if not os.path.exists(result_path_now):
                os.makedirs(result_path_now)
            save_path = os.path.join(result_path_now, '{}_{}_{}_fake.png'.format(self.e, self.i, i + 1))
            save_image(self.de_norm(image_list.data), save_path, normalize=True)

    def test(self):
        # Load trained parameters
        G_path = os.path.join(self.snapshot_path, '{}_G.pth'.format(self.test_model))
        self.G.load_state_dict(torch.load(G_path))
        self.G.eval()
        time_total = 0
        for i, (img_A, img_B) in enumerate(self.data_loader_test):
            start = time.time()
            try:
                processed_img_A = Image.open(img_A[0])
                processed_img_B = Image.open(img_B[0])
                processed_img_A = preprocess_makeup_gan(processed_img_A)
                processed_img_B = preprocess_makeup_gan(processed_img_B)
                processed_org_A = [self.to_var(item, requires_grad=False) for item in processed_img_A]
                processed_ref_B = [self.to_var(item, requires_grad=False) for item in processed_img_B]
            except Exception as e:
                print(str(e))
                print('vis_test image_A is: ', img_A[0])
                print('vis_test image_B is: ', img_B[0])
                continue

            real_org = processed_org_A[0]
            real_ref = processed_ref_B[0]

            image_list = []
            image_list_0 = []
            image_list.append(real_org)
            image_list.append(real_ref)

            # Get makeup result
            fake_A = Solver_MakeupGAN.generate(processed_org_A[0], processed_ref_B[0], processed_org_A[1],
                                               processed_ref_B[1], generator=self.G)
            fake_B = Solver_MakeupGAN.generate(processed_ref_B[0], processed_org_A[0], processed_ref_B[1],
                                               processed_org_A[1], generator=self.G)

            rec_A = Solver_MakeupGAN.generate(fake_A, processed_org_A[0], processed_org_A[1],
                                              processed_org_A[1], generator=self.G)
            rec_B = Solver_MakeupGAN.generate(fake_B, processed_ref_B[0], processed_ref_B[1],
                                              processed_ref_B[1], generator=self.G)

            time_total += time.time() - start
            image_list.append(fake_A)
            image_list_0.append(fake_A)
            image_list.append(fake_B)
            image_list.append(rec_A)
            image_list.append(rec_B)

            image_list = torch.cat(image_list, dim=3)
            image_list_0 = torch.cat(image_list_0, dim=3)

            result_path_now = os.path.join(self.result_path, "multi")
            if not os.path.exists(result_path_now):
                os.makedirs(result_path_now)
            save_path = os.path.join(result_path_now, '{}_{}_{}_fake.png'.format(self.e, self.i, i + 1))
            save_image(self.de_norm(image_list.data), save_path, nrow=1, padding=0, normalize=True)
            result_path_now = os.path.join(self.result_path, "single")
            if not os.path.exists(result_path_now):
                os.makedirs(result_path_now)
            save_path_0 = os.path.join(result_path_now, '{}_{}_{}_fake_single.png'.format(self.e, self.i, i + 1))
            save_image(self.de_norm(image_list_0.data), save_path_0, nrow=1, padding=0, normalize=True)
            print('Translated test images and saved into "{}"..!'.format(save_path))
        print("average time : {}".format(time_total / len(self.data_loader_test)))

    @staticmethod
    def image_test(real_A, mask_A, real_B, mask_B, shade_alpha=1):
        G = makeup_gan.Generator()
        G.load_state_dict(torch.load(pwd + '/makeup_G.pth', map_location=torch.device('cpu')))
        G.eval()

        cur_prama = None
        with torch.no_grad():
            cur_prama = Solver_MakeupGAN.generate(real_A, real_B, mask_A, mask_B, ret=True, generator=G, mode='test')
            cur_prama_source = Solver_MakeupGAN.generate(real_A, real_A, mask_A, mask_A, ret=True,
                                                         generator=G, mode='test')
            shade_gamma = cur_prama[0] * shade_alpha + cur_prama_source[0] * (1 - shade_alpha)
            shade_beta = cur_prama[1] * shade_alpha + cur_prama_source[1] * (1 - shade_alpha)
            fake_A = Solver_MakeupGAN.generate(real_A, real_B, mask_A, mask_B, gamma=shade_gamma, beta=shade_beta,
                                               generator=G, mode='test')
        fake_A = data2img(fake_A)
        return fake_A

    @staticmethod
    def partial_test(real_A, mask_aug_A, real_B, mask_aug_B, real_C, mask_aug_C, mask2use,
                     shade_alpha=1):
        G = makeup_gan.Generator()
        G.load_state_dict(torch.load(pwd + '/makeup_G.pth', map_location=torch.device('cpu')))
        G.eval()
        with torch.no_grad():
            cur_prama_B = Solver_MakeupGAN.generate(real_A, real_B, mask_aug_A, mask_aug_B, ret=True, generator=G,
                                                    mode='test')
            cur_prama_C = Solver_MakeupGAN.generate(real_A, real_C, mask_aug_A, mask_aug_C, ret=True, generator=G,
                                                    mode='test')

            cur_prama_source = Solver_MakeupGAN.generate(real_A, real_A, mask_aug_A, mask_aug_A, ret=True, generator=G,
                                                         mode='test')

            partial_gamma = cur_prama_B[0] * mask2use + cur_prama_C[0] * (1 - mask2use)
            partial_beta = cur_prama_B[1] * mask2use + cur_prama_C[1] * (1 - mask2use)

            partial_gamma = partial_gamma * shade_alpha + cur_prama_source[0] * (1 - shade_alpha)
            partial_beta = partial_beta * shade_alpha + cur_prama_source[1] * (1 - shade_alpha)

            fake_A = Solver_MakeupGAN.generate(real_A, real_B, mask_aug_A, mask2use, gamma=partial_gamma,
                                               beta=partial_beta, ret=False,
                                               generator=G, mode='test')
        fake_A = data2img(fake_A)
        return fake_A
