from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST, SegRNN, CycleNet, \
    iTransformer, TimeXer, GTR, GTRDLinear, GTRPatchTST, GTRiTransformer, APEC
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

import os
import time


class Exp_APEC(Exp_Basic):
    def __init__(self, args):
        super(Exp_APEC, self).__init__(args)
        self.plugin = None

    # ── model / data ──────────────────────────────────────────────────────────

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer, 'Transformer': Transformer, 'Informer': Informer,
            'DLinear': DLinear, 'NLinear': NLinear, 'Linear': Linear,
            'PatchTST': PatchTST, 'SegRNN': SegRNN, 'CycleNet': CycleNet,
            'iTransformer': iTransformer, 'TimeXer': TimeXer,
            'GTR': GTR, 'GTRDLinear': GTRDLinear, 'GTRPatchTST': GTRPatchTST,
            'GTRiTransformer': GTRiTransformer,
        }
        model = model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _build_plugin(self):
        if self.plugin is not None:
            return self.plugin
        n_channels = 1 if self.args.features == 'MS' else self.args.enc_in
        self.plugin = APEC.TrendPlugin(
            pred_len=self.args.pred_len,
            n_channels=n_channels,
        ).to(self.device)
        return self.plugin

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _target_offset(self):
        return -1 if self.args.features == 'MS' else 0

    def _target_slice(self, batch_y):
        return batch_y[:, -self.args.pred_len:, self._target_offset():]

    # ── backbone forward ──────────────────────────────────────────────────────

    def _forward_backbone(self, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle):
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).to(self.device)
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                out = self._dispatch_backbone(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle)
        else:
            out = self._dispatch_backbone(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle)
        return out[:, -self.args.pred_len:, self._target_offset():]

    def _dispatch_backbone(self, batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle):
        if any(s in self.args.model for s in ('CycleNet', 'GTR')):
            return self.model(batch_x, batch_cycle)
        if any(s in self.args.model for s in ('Linear', 'MLP', 'SegRNN', 'TST')):
            return self.model(batch_x)
        if self.args.output_attention:
            return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

    # ── backbone training (unchanged) ─────────────────────────────────────────

    def _vali_backbone(self, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)
                outputs = self._forward_backbone(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle)
                true = self._target_slice(batch_y)
                total_loss.append(criterion(outputs.detach().cpu(), true.detach().cpu()).item())
        self.model.train()
        return np.average(total_loss)

    def _train_backbone(self, setting, train_loader, vali_loader, path):
        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        criterion = nn.MSELoss()
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim, steps_per_epoch=train_steps,
                                             pct_start=self.args.pct_start, epochs=self.args.train_epochs,
                                             max_lr=self.args.learning_rate)
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)
                outputs = self._forward_backbone(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle)
                true = self._target_slice(batch_y)
                loss = criterion(outputs, true)
                train_loss.append(loss.item())
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()
            print("Backbone Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self._vali_backbone(vali_loader, criterion)
            print("Backbone Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Backbone early stopping")
                break
            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
        self.model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth')))

    # ── plugin training ───────────────────────────────────────────────────────

    def _train_plugin(self, train_loader, path):
        """Train T on the training set with backbone frozen.
        T converges to mean(y_true - y_hat) per (horizon, channel)."""
        optimizer = optim.Adam(self.plugin.parameters(), lr=self.args.apec_learning_rate)
        best_path = os.path.join(path, 'apec_plugin.pth')
        best_loss = None

        for epoch in range(self.args.apec_epochs):
            self.model.eval()
            self.plugin.train()
            train_loss = []
            epoch_time = time.time()

            for batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle in train_loader:
                optimizer.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)
                with torch.no_grad():
                    y_hat = self._forward_backbone(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle)
                true = self._target_slice(batch_y)
                pred = self.plugin(y_hat)
                loss = torch.mean((true - pred) ** 2)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())

            avg_loss = np.average(train_loss)
            T = self.plugin.T.detach()
            print("T Epoch: {0} | cost time: {1:.3f}s | Loss: {2:.7f} | "
                  "T mean={3:.5f}  std={4:.5f}  max|T|={5:.5f}".format(
                epoch + 1, time.time() - epoch_time, avg_loss,
                T.mean().item(), T.std().item(), T.abs().max().item()))

            if best_loss is None or avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.plugin.state_dict(), best_path)

        self.plugin.load_state_dict(torch.load(best_path))
        print("T training done. Final T: mean={:.5f}  std={:.5f}  max|T|={:.5f}".format(
            self.plugin.T.detach().mean().item(),
            self.plugin.T.detach().std().item(),
            self.plugin.T.detach().abs().max().item()))

    # ── train / test ──────────────────────────────────────────────────────────

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        _, vali_loader = self._get_data(flag='val')
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        print(">>>>>>>stage 1: train backbone : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting))
        self._train_backbone(setting, train_loader, vali_loader, path)

        for param in self.model.parameters():
            param.requires_grad = False
        self._build_plugin()

        print(">>>>>>>stage 2: train T on training set<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        self._train_plugin(train_loader, path)
        return self.model

    def test(self, setting, test=0):
        _, test_loader = self._get_data(flag='test')
        path = os.path.join(self.args.checkpoints, setting)
        self._build_plugin()

        if test:
            print('loading backbone and T')
            self.model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth')))
            self.plugin.load_state_dict(torch.load(os.path.join(path, 'apec_plugin.pth')))

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        preds, base_preds, trues = [], [], []
        self.model.eval()
        self.plugin.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)
                y_hat = self._forward_backbone(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle)
                true = self._target_slice(batch_y)
                pred = self.plugin(y_hat)
                preds.append(pred.detach().cpu().numpy())
                base_preds.append(y_hat.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())
                if i % 20 == 0:
                    input_x = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input_x[0, :, -1], true.detach().cpu().numpy()[0, :, -1]))
                    pd = np.concatenate((input_x[0, :, -1], pred.detach().cpu().numpy()[0, :, -1]))
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop(self.model, (batch_x.shape[1], batch_x.shape[2]))
            exit()

        preds      = np.concatenate(preds,      axis=0)
        base_preds = np.concatenate(base_preds, axis=0)
        trues      = np.concatenate(trues,      axis=0)

        mae, mse, *_       = metric(preds,      trues)
        base_mae, base_mse, *_ = metric(base_preds, trues)

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print('base_mse:{}, base_mae:{}'.format(base_mse, base_mae))
        print('apec_mse:{}, apec_mae:{}'.format(mse, mae))
        with open("result.txt", 'a') as f:
            f.write(setting + "  \n")
            f.write('base_mse:{}, base_mae:{}\n'.format(base_mse, base_mae))
            f.write('apec_mse:{}, apec_mae:{}\n\n'.format(mse, mae))

        np.save(folder_path + 'metrics_apec.npy', np.array([mae, mse]))
        np.save(folder_path + 'pred_apec.npy', preds)
        np.save(folder_path + 'pred_base.npy', base_preds)
        np.save(folder_path + 'true.npy', trues)
