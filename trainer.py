# -*- coding: utf-8 -*-

import time
import datetime

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import tqdm

import utils
from utils import AverageMeter


class Trainer(object):

    def __init__(
        self,
        cmd,
        cuda,
        model,
        optim=None,
        train_loader=None,
        valid_loader=None,
        test_loader=None,
        log_file=None,
        interval_validate=1,
        lr_scheduler=None,
        start_step=0,
        total_steps=1e5,
        beta=0.05,
        start_epoch=0,
        total_anneal_steps=200000,
        anneal_cap=0.2,
        do_normalize=True,
        checkpoint_dir=None,
        result_dir=None,
        print_freq=1,
        result_save_freq=1,
        checkpoint_freq=1,
    ):

        self.cmd = cmd
        self.cuda = cuda
        self.model = model
        print(f"Model has {utils.count_parameters(self.model)} parameters")

        self.optim = optim
        self.lr_scheduler = lr_scheduler

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.timestamp_start = datetime.datetime.now()

        if self.cmd == 'train':
            self.interval_validate = interval_validate

        self.start_step = start_step
        self.step = start_step
        self.total_steps = total_steps
        self.epoch = start_epoch

        self.do_normalize = do_normalize
        self.print_freq = print_freq
        self.checkpoint_freq = checkpoint_freq

        self.checkpoint_dir = checkpoint_dir

        self.total_anneal_steps = total_anneal_steps
        self.anneal_cap = anneal_cap

        self.n20_all = []
        self.ndcg_max_va, self.ap_max_va, self.n20_max_va, self.n100_max_va, self.r20_max_va, self.r50_max_va = 0, 0, 0, 0, 0, 0
        self.ndcg_max_te, self.ap_max_te, self.n20_max_te, self.n100_max_te, self.r20_max_te, self.r50_max_te = 0, 0, 0, 0, 0, 0
        self.ndcg_max_tr, self.ap_max_tr, self.n20_max_tr, self.n100_max_tr, self.r20_max_tr, self.r50_max_tr = 0, 0, 0, 0, 0, 0

        self.save_test_metric = False

        self.scaler = torch.cuda.amp.GradScaler()

    @torch.no_grad()
    def validate(self, cmd="valid"):
        assert cmd in ['valid', 'test', 'train']
        data_time = AverageMeter()
        self.model.eval()

        end = time.time()

        n20_list, n100_list, r20_list, r50_list = [], [], [], []
        ndcg_list, ap_list = [], []

        if cmd == 'valid':
            loader_ = self.valid_loader
        elif cmd == 'test':
            loader_ = self.test_loader
        elif cmd == 'train':
            loader_ = self.train_loader

        step_counter = 0
        for batch_idx, (data_tr, data_te, prof) in tqdm.tqdm(
            enumerate(loader_),
            total=len(loader_),
            desc='{} check epoch={}, len={}'.format('Valid' if cmd == 'valid' else 'Test', self.epoch, len(loader_)),
            ncols=80,
            leave=False,
            position=1,
        ):
            if cmd == 'train':
                data_te = data_tr

            step_counter = step_counter + 1

            if self.cuda:
                data_tr = data_tr.cuda()
                data_te = data_te.cuda()
                prof = prof.cuda()
            data_tr = Variable(data_tr)
            prof = Variable(prof)
            data_time.update(time.time() - end)
            end = time.time()

            with torch.cuda.amp.autocast(enabled=False):
                if self.model.__class__.__name__ == 'MultiVAE':
                    logits, KL, mu_q, std_q, epsilon, sampled_z = self.model.forward(data_tr, prof)
                else:
                    logits = self.model.forward(data_tr)
            pred_val = logits
            if cmd != 'train':
                pred_val[data_tr.bool()] = -float('inf')

            ap_list.append(utils.average_precision(pred_val, data_te))
            ndcg_list.append(utils.NDCG_binary_at_k_batch(pred_val, data_te, None))
            n20_list.append(utils.NDCG_binary_at_k_batch(pred_val, data_te, k=20))
            n100_list.append(utils.NDCG_binary_at_k_batch(pred_val, data_te, k=100))
            r20_list.append(utils.Recall_at_k_batch(pred_val, data_te, k=20))
            r50_list.append(utils.Recall_at_k_batch(pred_val, data_te, k=50))

        ap_list = np.concatenate(ap_list, axis=0)
        ndcg_list = np.concatenate(ndcg_list, axis=0)
        n20_list = np.concatenate(n20_list, axis=0)
        n100_list = np.concatenate(n100_list, axis=0)
        r20_list = np.concatenate(r20_list, axis=0)
        r50_list = np.concatenate(r50_list, axis=0)

        if cmd == 'valid':
            self.ndcg_max_va = max(self.ndcg_max_va, ndcg_list.mean())
            self.ap_max_va = max(self.ap_max_va, ap_list.mean())
            self.n20_max_va = max(self.n20_max_va, n20_list.mean())
            self.n100_max_va = max(self.n100_max_va, n100_list.mean())
            self.save_test_metric = r20_list.mean() >= self.r20_max_va
            self.r20_max_va = max(self.r20_max_va, r20_list.mean())
            self.r50_max_va = max(self.r50_max_va, r50_list.mean())
            max_metrics = "{},{},{},{:.5f},{:.5f},{:.5f},{:.5f}".format(cmd, self.epoch, self.step, self.ndcg_max_va, self.ap_max_va, self.r20_max_va, self.n20_max_va)

        elif cmd == 'test':
            if r20_list.mean() >= self.r20_max_te:
                # self.n20_max_te = max(self.n20_max_te, n20_list.mean())
                # self.n100_max_te = max(self.n100_max_te, n100_list.mean())
                # self.r20_max_te = max(self.r20_max_te, r20_list.mean())
                # self.r50_max_te = max(self.r50_max_te, r50_list.mean())
                self.ndcg_max_te = ndcg_list.mean()
                self.ap_max_te = ap_list.mean()
                self.n20_max_te = n20_list.mean()
                self.n100_max_te = n100_list.mean()
                self.r20_max_te = r20_list.mean()
                self.r50_max_te = r50_list.mean()
                self.save_test_metric = False
            max_metrics = "{},{},{},{:.5f},{:.5f},{:.5f},{:.5f}".format(cmd, self.epoch, self.step, self.ndcg_max_te, self.ap_max_te, self.r20_max_te, self.n20_max_te)

        elif cmd == 'train':
            self.ndcg_max_tr = max(self.ndcg_max_tr, ndcg_list.mean())
            self.ap_max_tr = max(self.ap_max_tr, ap_list.mean())
            self.n20_max_tr = max(self.n20_max_tr, n20_list.mean())
            self.n100_max_tr = max(self.n100_max_tr, n100_list.mean())
            self.r20_max_tr = max(self.r20_max_tr, r20_list.mean())
            self.r50_max_tr = max(self.r50_max_tr, r50_list.mean())
            max_metrics = "{},{},{},{:.5f},{:.5f},{:.5f},{:.5f}".format(cmd, self.epoch, self.step, self.ndcg_max_tr, self.ap_max_tr, self.r20_max_tr, self.n20_max_tr)

        metrics = []
        metrics.append(max_metrics)
        metrics.append("NDCG,{:.5f},{:.5f}".format(np.mean(ndcg_list), np.std(ndcg_list) / np.sqrt(len(ndcg_list))))
        metrics.append("AP,{:.5f},{:.5f}".format(np.mean(ap_list), np.std(ap_list) / np.sqrt(len(ap_list))))
        metrics.append("Recall@20,{:.5f},{:.5f}".format(np.mean(r20_list), np.std(r20_list) / np.sqrt(len(r20_list))))
        metrics.append("NDCG@20,{:.5f},{:.5f}".format(np.mean(n20_list), np.std(n20_list) / np.sqrt(len(n20_list))))
        print('\n' + ",".join(metrics))

        self.model.train()

    def train_epoch(self):
        data_time = AverageMeter()
        losses = AverageMeter()
        self.model.train()

        end = time.time()
        iterator = tqdm.tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc='Train check epoch={}, len={}'.format(self.epoch, len(self.train_loader)),
            # ncols=80,
            leave=False,
            position=1,
        )

        for batch_idx, (data_tr, data_te, prof) in iterator:
            self.step += 1

            if self.cuda:
                data_tr = data_tr.cuda()
                prof = prof.cuda()
            data_tr = Variable(data_tr)
            prof = Variable(prof)
            data_time.update(time.time() - end)
            end = time.time()

            with torch.cuda.amp.autocast(enabled=False):
                if self.model.__class__.__name__ == 'MultiVAE':
                    logits, KL, mu_q, std_q, epsilon, sampled_z = self.model.forward(data_tr, prof)
                else:
                    logits = self.model.forward(data_tr)

                log_softmax_var = F.log_softmax(logits, dim=1)
                neg_ll = - torch.mean(torch.sum(log_softmax_var * data_tr, dim=1))
                l2_reg = self.model.get_l2_reg()

                if self.model.__class__.__name__ == 'MultiVAE':
                    if self.total_anneal_steps > 0:
                        self.anneal = min(self.anneal_cap, 1. * self.step / self.total_anneal_steps)
                    else:
                        self.anneal = self.anneal_cap

                    loss = neg_ll + self.anneal * KL + l2_reg
                    # print("MultiVAE", self.epoch, batch_idx, loss.item(), neg_ll.cpu().detach().numpy(), KL.cpu().detach().numpy(), l2_reg.cpu().detach().numpy() / 2, self.anneal, self.step, self.optim.param_groups[0]['lr'])
                else:
                    loss = neg_ll + l2_reg
                    # print("MultiDAE", self.epoch, batch_idx, loss.item(), neg_ll.cpu().detach().numpy(), l2_reg.cpu().detach().numpy() / 2, self.step)

            # backprop
            self.model.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()

            losses.update(loss.item())
            iterator.set_postfix({'loss': neg_ll.item(), 'avg_loss': losses.avg})

            # if self.interval_validate > 0 and (self.step + 1) % self.interval_validate == 0:
            #     print("CALLING VALID", cmd, self.step, )
            #     self.validate()

    def train(self):
        max_epoch = 200
        for epoch in tqdm.trange(0, max_epoch, desc='Train', ncols=80, position=0):
            self.epoch = epoch
            self.train_epoch()
            self.lr_scheduler.step()
            self.validate(cmd='valid')
            self.validate(cmd='test')
            # self.validate(cmd='train')
