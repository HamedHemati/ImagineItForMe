import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
from time import time
import os
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import NetD, NetG
from cub2011 import get_cub2011


class Trainer():
    """
    Trainer
    """
    def __init__(self, args):
        # training parameters
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.iter_d = args.iter_d
        self.lr = args.lr
        self.b1 = args.b1
        self.b2 = args.b2
        self.lamb = args.lamb
        self.lambp = args.lambp
        self.cuda = args.cuda
        self.continue_traning = args.continue_training
        self.netd_checkpoint = args.netd_checkpoint
        self.netg_checkpoint = args.netg_checkpoint
        self.use_valset = args.use_valset
        self.num_workers = args.num_workers
        self.trainset_loader = None
        self.valset_loader = None
        self.tf_idf = None
        self.ds_name = args.ds_name
        self.ds_path = args.ds_path
        self.save_every = args.save_every
        self.save_dir = args.save_dir
        self.n_cls = args.n_cls
        self.n_z = args.n_z
        self.n_feat = args.n_feat
        self.n_tfidf = args.n_tfidf
        self.n_emb = args.n_emb
        self.n_mdl = args.n_mdl
        # model and loss functions
        self.net_d = NetD(n_feat=self.n_feat, n_mdl=self.n_mdl, num_cls=self.n_cls)
        self.net_g = NetG(n_tfidf=self.n_tfidf, n_emb=self.n_emb, n_z=self.n_z, n_mdl=self.n_mdl, n_feat=self.n_feat)
        self.bce_loss = nn.BCELoss()
        self.nll_loss = nn.NLLLoss()
        if self.cuda and torch.cuda.is_available():
            print("CUDA is enabled")
            self.net_d = self.net_d.cuda()
            self.net_g = self.net_g.cuda()
            self.bce_loss = self.bce_loss.cuda()
            self.nll_loss = self.nll_loss.cuda()
        # optimizers
        self.optimizer_d = optim.Adam(params=self.net_d.parameters(),lr=self.lr,betas=(self.b1, self.b2))
        self.optimizer_g = optim.Adam(params=self.net_g.parameters(),lr=self.lr,betas=(self.b1, self.b2))
        # directories and log file
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(os.path.join(self.save_dir,'netd_checkpoints')):
            os.makedirs(os.path.join(self.save_dir,'netd_checkpoints'))
        if not os.path.exists(os.path.join(self.save_dir,'netg_checkpoints')):            
            os.makedirs(os.path.join(self.save_dir,'netg_checkpoints')) 
        if not os.path.exists(os.path.join(self.save_dir,'samples')):            
            os.makedirs(os.path.join(self.save_dir,'samples'))
        self.create_log_file()
        # load the dataset
        train_set,val_set,_,self.tf_idf = get_cub2011(ds_path=self.ds_path, val=self.use_valset)
        self.trainset_loader = DataLoader(dataset=train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        # check if validation set is used
        if self.use_valset:
            self.valset_loader = DataLoader(dataset=val_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        print('loaded the dataset successfuly')
        # check if it needs to load checkpoints for continuing training
        if self.continue_traning:
            self.load_checkpoints()


    def train(self):
        """
        Starts training procedure
        """
        netd_losses = []
        netg_losses = []
        val_losses = []
        for epoch in range(self.epochs):
            loss_d, loss_g = self.train_epoch(epoch)
            netd_losses.append(loss_d)
            netg_losses.append(loss_g)

            self.save_plot(netd_losses, netd_losses)
            self.save_checkpoints(epoch)



    def train_epoch(self, epoch):
        """
        Tranining Epoch
        """
        self.net_d.train()
        self.net_g.train()       
        netd_loss_sum = 0
        netg_loss_sum = 0
        print("\nEpoch %d"%(epoch))
        start_time = time()
        for i,(cnn_feats, clss) in enumerate(self.trainset_loader):
            batch_size = cnn_feats.size(0)
            curr_clss = clss
            cnn_feats, clss = Variable(cnn_feats), Variable(clss)
            lbl_real, lbl_fake = Variable(torch.ones(batch_size,1)), Variable(torch.zeros(batch_size,1))
            if self.cuda:
                cnn_feats, clss = cnn_feats.cuda(), clss.cuda()
                lbl_real, lbl_fake = lbl_real.cuda(), lbl_fake.cuda()  
            
            self.net_g.zero_grad()
            # update DISCRIMINATOR parameters
            lossd_mean = 0
            for t in range(self.iter_d):
                perm_ind = torch.randperm(batch_size) # random permutation of indices
                if self.cuda:
                    perm_ind = perm_ind.cuda
                #???? does rand perm really affect the training?!
                # loss for fake samples
                rnd_noise = Variable(torch.rand(batch_size, self.n_z))
                rnd_tfidf = Variable(self.tf_idf[curr_clss[perm_ind] ])
                if self.cuda:
                    noise = noise.cuda()
                fake_clss = clss[perm_ind]   
                fake_samples = self.net_g(rnd_noise, rnd_tfidf)
                if self.cuda:
                    fake_clss, fake_samples = fake_clss.cuda(), fake_samples.cuda()
                outd_fake_s, outd_fake_cls = self.net_d(fake_samples.detach())
                lossd_fake_s = self.bce_loss(outd_fake_s, lbl_fake)
                lossd_fake_cls = self.nll_loss(outd_fake_cls, fake_clss)
                
                # loss for real samples
                outd_real_s, outd_real_cls = self.net_d(cnn_feats)
                lossd_real_s = self.bce_loss(outd_real_s, lbl_real)
                lossd_real_cls = self.nll_loss(outd_real_cls, clss)
                
                # compute L_GP by interpolating synthesized and real features
                gradient_penalty =  self.calculate_gradient_penalty(fake_samples, cnn_feats)
                
                # compute gradient of netD and update the weights
                loss_d = lossd_fake_s - lossd_real_s + gradient_penalty + 0.5*(lossd_real_cls+lossd_fake_cls)
                loss_d.backward()
                self.optimizer_d.step()
                lossd_mean += loss_d.data[0]

            netd_loss_sum += lossd_mean / self.iter_d

            # update GENERATOR parameters

            # generate fake samples
            perm_ind = torch.randperm(batch_size) # random permutation of indices
            if self.cuda:
                perm_ind = perm_ind.cuda
            # TODO does rand perm really affect the training?!
            # loss for fake samples
            rnd_noise = Variable(torch.rand(batch_size, self.n_z))
            rnd_tfidf = Variable(self.tf_idf[curr_clss[perm_ind] ])
            if self.cuda:
                noise = noise.cuda()
            fake_clss = clss[perm_ind]   
            fake_samples = self.net_g(rnd_noise, rnd_tfidf)
            if self.cuda:
                fake_clss, fake_samples = fake_clss.cuda(), fake_samples.cuda()
            outd_fake_s, outd_fake_cls = self.net_d(fake_samples)
            lossg_fake_s = self.bce_loss(outd_fake_s, lbl_fake)
            lossg_fake_cls = self.nll_loss(outd_fake_cls, fake_clss)
            loss_g_f = -lossg_fake_s + lossg_fake_cls
            
            # compute mean set error
            clss_set = {i:None for i in range(self.n_cls)}
            for j in range(batch_size):
                cl = curr_clss[perm_ind][j]
                if clss_set[cl] is not None:
                    clss_set[cl] = torch.cat([clss_set[cl], fake_samples[j]], 0)
                else:
                    clss_set[cl] = fake_samples[j]

            # compute sum of means
            mean_fakes = fake_samples.mean(dim=0)
            coef = self.lambp * (1.0/self.n_cls)
            loss_reg = 0
            for j in range(self.n_cls):
                if clss_set[j] is not None:
                    loss_reg += torch.norm(clss_set[j].mean(dim=0) - mean_fakes)
            loss_reg = coef * loss_reg
            loss_g = loss_g_f + loss_reg
            loss_g.backward()
            self.optimizer_g.step()            
            
            netg_loss_sum += loss_g.data[0]
            
            print("%.2f%% of epoch %d is done , loss_d: %f, loss_g: %f" 
                   %(float(i)/len(self.trainset_loader)*100, epoch, netd_loss_sum, netg_loss_sum))
            
        return netd_loss_sum/len(self.trainset_loader), netg_loss_sum/len(self.trainset_loader)


    def calculate_gradient_penalty(self, fake_samples, real_samples):
        """
        Calculuates gradient penalty
        """
        alpha = random.random() #---> find a good hyperparamether
        interp = alpha*fake_samples + (1-alpha)*real_samples
        outd_interp_s, outd_interp_cls = self.net_d(interp)
        # for Lipschitz constraint we need to compute the gradient with respect to interpolated features
        grad_out1 = torch.ones(outd_interp_s.size())
        grad_out2 = torch.ones(outd_interp_cls.size())
        if self.cuda:
            grad_out1, grad_out2 = grad_out1.cuda(), grad_out2.cuda()
        gradients = autograd.grad(outputs=[outd_interp_s, outd_interp_cls], inputs=interp, grad_outputs=[grad_out1, grad_out2],
                             create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lamb                              
        return gradient_penalty


    def save_plot(self, netd_losses=[], netg_losses=[], val_losses=[]):
        """
        Saves plot for each epoch
        """
        plt.plot(netd_losses, color='red', label='NetD Loss')
        plt.plot(netg_losses, color='blue', label='NetG Loss')
        plt.plot(val_losses, color='green', label='Val Loss')
        plt.xlabel('epoch')
        plt.ylabel('error')
        plt.legend(loc='best')
        plt.savefig(os.path.join(self.save_dir,'loss_graph.png'))
        plt.close()


    def save_checkpoints(self, epoch):
        """
        Saves checkpoints for current epoch
        """
        if epoch%self.save_every==0:
            name_netd = "netd_checkpoints/netd_epoch_" + str(epoch) + ".pth"
            name_netg = "netg_checkpoints/netg_epoch_" + str(epoch) + ".pth"
            torch.save(self.net_d.state_dict(), os.path.join(self.save_dir, name_netd))
            torch.save(self.net_g.state_dict(), os.path.join(self.save_dir, name_netg))
            print("checkpoints for epoch %d saved successfuly" %(epoch))


    def load_checkpoints(self):
        """
        Loades checkpoints for conituing training
        """
        self.net_d.load_state_dict(torch.load(self.netd_checkpoint))
        self.net_g.load_state_dict(torch.load(self.netg_checkpoint))
        print('checkpoints loaded successfuly\n')
    

    def create_log_file(self):
        """
        Creates log file
        """
        # create a log file and initialize it with training parameters info
        log_msg = '********************************************\n'
        log_msg += '            Training Parameters\n'
        log_msg += 'Dataset:%s\n'%(self.ds_name)
        log_msg += 'Number of classes:%d\n'%(self.n_cls)  
        log_msg += 'Batch size:%d\n'%(self.batch_size)
        log_msg += 'Number of epochs:%d\n'%(self.epochs)
        log_msg += 'lr:%f, b1:%f, b2:%f\n'%(self.lr, self.b1, self.b2)
        log_msg += 'n-z:%d, n-feat:%d, n-tfidf:%d, n-emb:%d, n-mdl:%d\n'%(self.n_z, self.n_feat, self.n_tfidf, self.n_emb, self.n_mdl)
        log_msg += '********************************************\n'
        print(log_msg)
        with open(os.path.join(self.save_dir, 'training_log.txt'),'a') as log_file:
            log_file.write(log_msg) 
