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
from torch.utils.data import DataLoader, Dataset, Subset

import os
import time


class APECWindowDataset(Dataset):
    def __init__(self, base_data, indices, residuals, window, feature_offset):
        self.base_data = base_data
        self.indices = list(indices)
        self.residuals = residuals
        self.window = window
        self.feature_offset = feature_offset

    def __len__(self):
        return len(self.indices)

    def _left_pad(self, values):
        if len(values) >= self.window:
            return values
        pad_shape = (self.window - len(values), values.shape[-1])
        pad = np.zeros(pad_shape, dtype=np.float32)
        return np.concatenate([pad, values], axis=0)

    def __getitem__(self, item):
        index = self.indices[item]
        seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index = self.base_data[index]
        s_end = index + self.base_data.seq_len
        start = max(0, s_end - self.window)

        x_win = self.base_data.data_x[start:s_end, self.feature_offset:].astype(np.float32)
        e_win = self.residuals[start:s_end].astype(np.float32)
        x_win = self._left_pad(x_win)
        e_win = self._left_pad(e_win)

        return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index, x_win, e_win


class Exp_APEC(Exp_Basic):
    def __init__(self, args):
        super(Exp_APEC, self).__init__(args)
        self.plugin = APEC.ChannelIndependentPlugIn(
            window=args.apec_window,
            pred_len=args.pred_len,
            d_model=args.apec_hidden,
            dropout=args.apec_dropout,
        ).to(self.device)
        self.q = None
        self.gamma = 1.0

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
            'SegRNN': SegRNN,
            'CycleNet': CycleNet,
            'iTransformer': iTransformer,
            'TimeXer': TimeXer,
            'GTR': GTR,
            'GTRDLinear': GTRDLinear,
            'GTRPatchTST': GTRPatchTST,
            'GTRiTransformer': GTRiTransformer
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _target_offset(self):
        return -1 if self.args.features == 'MS' else 0

    def _target_slice(self, batch_y):
        f_dim = self._target_offset()
        return batch_y[:, -self.args.pred_len:, f_dim:]

    def _make_loader(self, dataset, shuffle=False, drop_last=False):
        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            num_workers=self.args.num_workers,
            drop_last=drop_last,
        )

    def _forward_backbone(self, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle):
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self._dispatch_backbone(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle)
        else:
            outputs = self._dispatch_backbone(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle)

        f_dim = self._target_offset()
        return outputs[:, -self.args.pred_len:, f_dim:]

    def _dispatch_backbone(self, batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle):
        if any(substr in self.args.model for substr in {'CycleNet', 'GTR'}):
            return self.model(batch_x, batch_cycle)
        if any(substr in self.args.model for substr in {'Linear', 'MLP', 'SegRNN', 'TST'}):
            return self.model(batch_x)
        if self.args.output_attention:
            return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

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
                loss = criterion(outputs.detach().cpu(), true.detach().cpu())
                total_loss.append(loss.item())
        self.model.train()
        return np.average(total_loss)

    def _split_train_indices(self, train_len):
        backbone_end = int(train_len * self.args.apec_backbone_ratio)
        plugin_end = int(train_len * (self.args.apec_backbone_ratio + self.args.apec_plugin_ratio))
        backbone_end = max(2, min(backbone_end, train_len - 2))
        plugin_end = max(backbone_end + 1, min(plugin_end, train_len - 1))
        plugin_start = min(backbone_end + self.args.apec_window, train_len - 2)
        plugin_end = min(max(plugin_start + 2, plugin_end), train_len - 1)
        plugin_len = plugin_end - plugin_start
        plugin_val_size = max(1, int(plugin_len * self.args.apec_plugin_val_ratio))
        plugin_val_size = min(plugin_val_size, plugin_len - 1)
        plugin_train_end = plugin_end - plugin_val_size

        backbone_val_size = max(1, int(backbone_end * 0.15))
        backbone_train_end = max(1, backbone_end - backbone_val_size)

        return (
            range(0, backbone_train_end),
            range(backbone_train_end, backbone_end),
            range(plugin_start, plugin_train_end),
            range(plugin_train_end, plugin_end),
            range(plugin_end, train_len),
        )

    def _train_backbone(self, setting, train_loader, vali_loader, path):
        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        criterion = nn.MSELoss()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=train_steps,
            pct_start=self.args.pct_start,
            epochs=self.args.train_epochs,
            max_lr=self.args.learning_rate,
        )

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

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))

    def _build_one_step_residuals(self, data_set):
        feature_offset = self._target_offset()
        channels = data_set.data_x[:, feature_offset:].shape[-1]
        residuals = np.zeros((len(data_set.data_x), channels), dtype=np.float32)
        loader = self._make_loader(data_set, shuffle=False, drop_last=False)

        self.model.eval()
        start = 0
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle in loader:
                batch_size = batch_x.shape[0]
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                outputs = self._forward_backbone(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle)
                true = self._target_slice(batch_y)
                err = (true[:, 0, :] - outputs[:, 0, :]).detach().cpu().numpy().astype(np.float32)
                positions = np.arange(start, start + batch_size) + data_set.seq_len
                valid = positions < len(residuals)
                residuals[positions[valid]] = err[valid]
                start += batch_size
        return residuals

    def _set_logvar_trainable(self, trainable):
        for param in self.plugin.head_logvar.parameters():
            param.requires_grad = trainable

    def _apec_loss(self, y, y_hat, delta, logvar, epoch):
        pred = y_hat + delta
        mse = torch.mean((y - pred) ** 2)
        if epoch < self.args.apec_var_warmup:
            return mse, mse.detach(), torch.zeros_like(mse.detach())

        logvar = logvar.clamp(self.args.apec_logvar_min, self.args.apec_logvar_max)
        var = torch.exp(logvar)
        nll = 0.5 * (logvar + (y - pred) ** 2 / (var + 1e-6))
        nll = nll.mean()
        return mse + self.args.apec_nll_weight * nll, mse.detach(), nll.detach()

    def _eval_plugin_mse(self, eval_loader, gamma=1.0):
        corrected_losses = []
        base_losses = []
        self.model.eval()
        self.plugin.eval()
        with torch.no_grad():
            for batch in eval_loader:
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle, x_win, e_win = batch
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)
                x_win = x_win.float().to(self.device)
                e_win = e_win.float().to(self.device)

                y_hat = self._forward_backbone(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle)
                true = self._target_slice(batch_y)
                delta, _ = self.plugin(x_win, e_win, y_hat)
                pred = y_hat + gamma * delta
                corrected_losses.append(torch.mean((true - pred) ** 2).item())
                base_losses.append(torch.mean((true - y_hat) ** 2).item())
        return np.average(corrected_losses), np.average(base_losses)

    def _select_gamma(self, eval_loader):
        if self.args.apec_gamma_step <= 0:
            self.gamma = 1.0
            return self.gamma

        gamma_values = np.arange(0.0, 1.0 + 0.5 * self.args.apec_gamma_step, self.args.apec_gamma_step)
        best_gamma = 0.0
        best_mse = None
        for gamma in gamma_values:
            mse, _ = self._eval_plugin_mse(eval_loader, gamma=float(gamma))
            if best_mse is None or mse < best_mse:
                best_mse = mse
                best_gamma = float(gamma)
        self.gamma = best_gamma
        return self.gamma

    def _train_plugin(self, plugin_loader, plugin_val_loader, path):
        optimizer = optim.Adam(self.plugin.parameters(), lr=self.args.apec_learning_rate)
        best_path = os.path.join(path, 'apec_plugin.pth')
        best_val_mse = None
        bad_epochs = 0

        for epoch in range(self.args.apec_epochs):
            self.model.eval()
            self.plugin.train()
            self._set_logvar_trainable(epoch >= self.args.apec_var_warmup)
            train_loss = []
            train_mse = []
            train_nll = []
            epoch_time = time.time()

            for batch in plugin_loader:
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle, x_win, e_win = batch
                optimizer.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)
                x_win = x_win.float().to(self.device)
                e_win = e_win.float().to(self.device)

                with torch.no_grad():
                    y_hat = self._forward_backbone(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle)
                true = self._target_slice(batch_y)
                delta, logvar = self.plugin(x_win, e_win, y_hat)
                loss, mse, nll = self._apec_loss(true, y_hat, delta, logvar, epoch)
                loss.backward()
                optimizer.step()

                train_loss.append(loss.item())
                train_mse.append(mse.item())
                train_nll.append(nll.item())

            print("APEC Epoch: {0} cost time: {1}".format(epoch + 1, time.time() - epoch_time))
            val_mse, base_val_mse = self._eval_plugin_mse(plugin_val_loader)
            print("APEC Epoch: {0} | Loss: {1:.7f} MSE: {2:.7f} NLL: {3:.7f}".format(
                epoch + 1, np.average(train_loss), np.average(train_mse), np.average(train_nll)))
            print("APEC Epoch: {0} | Val MSE: {1:.7f} Base Val MSE: {2:.7f}".format(
                epoch + 1, val_mse, base_val_mse))

            if best_val_mse is None or val_mse < best_val_mse:
                best_val_mse = val_mse
                bad_epochs = 0
                torch.save(self.plugin.state_dict(), best_path)
            else:
                bad_epochs += 1
                print("APEC early stopping counter: {} out of {}".format(
                    bad_epochs, self.args.apec_plugin_patience))
                if bad_epochs >= self.args.apec_plugin_patience:
                    print("APEC plug-in early stopping")
                    break

        self.plugin.load_state_dict(torch.load(best_path))
        gamma = self._select_gamma(plugin_val_loader)
        gamma_path = os.path.join(path, 'apec_gamma.pt')
        torch.save({'gamma': gamma}, gamma_path)
        val_mse, base_val_mse = self._eval_plugin_mse(plugin_val_loader, gamma=gamma)
        print("APEC selected gamma: {:.2f} | Shrunk Val MSE: {:.7f} Base Val MSE: {:.7f}".format(
            gamma, val_mse, base_val_mse))

    def _calibrate(self, cal_loader, path):
        scores = []
        self.model.eval()
        self.plugin.eval()
        with torch.no_grad():
            for batch in cal_loader:
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle, x_win, e_win = batch
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)
                x_win = x_win.float().to(self.device)
                e_win = e_win.float().to(self.device)

                y_hat = self._forward_backbone(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle)
                true = self._target_slice(batch_y)
                delta, logvar = self.plugin(x_win, e_win, y_hat)
                sigma = torch.exp(0.5 * logvar.clamp(self.args.apec_logvar_min, self.args.apec_logvar_max))
                score = torch.abs(true - (y_hat + self.gamma * delta)) / (sigma + 1e-6)
                scores.append(score.detach().cpu())

        scores = torch.cat(scores, dim=0)
        self.q = torch.quantile(scores, 1 - self.args.apec_alpha, dim=0)
        torch.save({'q': self.q, 'alpha': self.args.apec_alpha, 'gamma': self.gamma}, os.path.join(path, 'apec_q.pt'))
        print("APEC conformal q calibrated with shape {}".format(tuple(self.q.shape)))

    def train(self, setting):
        train_data, _ = self._get_data(flag='train')
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        backbone_train_idx, backbone_val_idx, plugin_idx, plugin_val_idx, cal_idx = self._split_train_indices(len(train_data))
        backbone_train_loader = self._make_loader(Subset(train_data, backbone_train_idx), shuffle=True, drop_last=False)
        backbone_val_loader = self._make_loader(Subset(train_data, backbone_val_idx), shuffle=False, drop_last=False)

        print("APEC split sizes | backbone_train: {} backbone_val: {} plugin: {} plugin_val: {} calibration: {}".format(
            len(backbone_train_idx), len(backbone_val_idx), len(plugin_idx), len(plugin_val_idx), len(cal_idx)))
        print(">>>>>>>stage 1: train frozen backbone : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting))
        self._train_backbone(setting, backbone_train_loader, backbone_val_loader, path)

        for param in self.model.parameters():
            param.requires_grad = False

        print(">>>>>>>stage 2: build one-step residuals<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        train_residuals = self._build_one_step_residuals(train_data)

        feature_offset = self._target_offset()
        plugin_data = APECWindowDataset(train_data, plugin_idx, train_residuals, self.args.apec_window, feature_offset)
        plugin_val_data = APECWindowDataset(train_data, plugin_val_idx, train_residuals, self.args.apec_window, feature_offset)
        cal_data = APECWindowDataset(train_data, cal_idx, train_residuals, self.args.apec_window, feature_offset)
        plugin_loader = self._make_loader(plugin_data, shuffle=True, drop_last=False)
        plugin_val_loader = self._make_loader(plugin_val_data, shuffle=False, drop_last=False)
        cal_loader = self._make_loader(cal_data, shuffle=False, drop_last=False)

        print(">>>>>>>stage 3: train APEC plug-in<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        self._train_plugin(plugin_loader, plugin_val_loader, path)
        print(">>>>>>>stage 4: conformal calibration<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        self._calibrate(cal_loader, path)
        return self.model

    def test(self, setting, test=0):
        test_data, _ = self._get_data(flag='test')
        path = os.path.join(self.args.checkpoints, setting)

        if test:
            print('loading APEC backbone, plug-in, and q')
            self.model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth')))
            self.plugin.load_state_dict(torch.load(os.path.join(path, 'apec_plugin.pth')))
            q_state = torch.load(os.path.join(path, 'apec_q.pt'))
            self.q = q_state['q']
            if 'gamma' in q_state:
                self.gamma = float(q_state['gamma'])
            else:
                self.gamma = float(torch.load(os.path.join(path, 'apec_gamma.pt'))['gamma'])

        if self.q is None:
            q_path = os.path.join(path, 'apec_q.pt')
            if os.path.exists(q_path):
                q_state = torch.load(q_path)
                self.q = q_state['q']
                if 'gamma' in q_state:
                    self.gamma = float(q_state['gamma'])
                else:
                    self.gamma = float(torch.load(os.path.join(path, 'apec_gamma.pt'))['gamma'])
            else:
                raise RuntimeError('APEC q is not calibrated. Run training before test.')
        print("APEC test gamma: {:.2f}".format(self.gamma))

        print(">>>>>>>APEC test: build test residuals<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        test_residuals = self._build_one_step_residuals(test_data)
        feature_offset = self._target_offset()
        test_apec_data = APECWindowDataset(
            test_data,
            range(len(test_data)),
            test_residuals,
            self.args.apec_window,
            feature_offset,
        )
        test_loader = self._make_loader(test_apec_data, shuffle=False, drop_last=False)

        preds = []
        base_preds = []
        trues = []
        lowers = []
        uppers = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        q = self.q.to(self.device)
        self.model.eval()
        self.plugin.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle, x_win, e_win = batch
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)
                x_win = x_win.float().to(self.device)
                e_win = e_win.float().to(self.device)

                y_hat = self._forward_backbone(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle)
                true = self._target_slice(batch_y)
                delta, logvar = self.plugin(x_win, e_win, y_hat)
                sigma = torch.exp(0.5 * logvar.clamp(self.args.apec_logvar_min, self.args.apec_logvar_max))
                pred = y_hat + self.gamma * delta
                lower = pred - q * sigma
                upper = pred + q * sigma

                preds.append(pred.detach().cpu().numpy())
                base_preds.append(y_hat.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())
                lowers.append(lower.detach().cpu().numpy())
                uppers.append(upper.detach().cpu().numpy())

                if i % 20 == 0:
                    input_x = batch_x.detach().cpu().numpy()
                    true_np = true.detach().cpu().numpy()
                    pred_np = pred.detach().cpu().numpy()
                    gt = np.concatenate((input_x[0, :, -1], true_np[0, :, -1]), axis=0)
                    pd = np.concatenate((input_x[0, :, -1], pred_np[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop(self.model, (batch_x.shape[1], batch_x.shape[2]))
            exit()

        preds = np.concatenate(preds, axis=0)
        base_preds = np.concatenate(base_preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        lowers = np.concatenate(lowers, axis=0)
        uppers = np.concatenate(uppers, axis=0)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        base_mae, base_mse, _, _, _, _, _ = metric(base_preds, trues)
        coverage = np.mean((trues >= lowers) & (trues <= uppers))
        mean_width = np.mean(uppers - lowers)
        winkler = self._winkler_score(trues, lowers, uppers, self.args.apec_alpha)

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print('base_mse:{}, base_mae:{}'.format(base_mse, base_mae))
        print('apec_mse:{}, apec_mae:{}, coverage:{}, width:{}, winkler:{}, gamma:{}'.format(
            mse, mae, coverage, mean_width, winkler, self.gamma))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('base_mse:{}, base_mae:{}\n'.format(base_mse, base_mae))
        f.write('apec_mse:{}, apec_mae:{}, coverage:{}, width:{}, winkler:{}, gamma:{}'.format(
            mse, mae, coverage, mean_width, winkler, self.gamma))
        f.write('\n\n')
        f.close()

        np.save(folder_path + 'metrics_apec.npy', np.array([mae, mse, coverage, mean_width, winkler, self.gamma]))
        np.save(folder_path + 'pred_apec.npy', preds)
        np.save(folder_path + 'pred_base.npy', base_preds)
        np.save(folder_path + 'true.npy', trues)
        np.save(folder_path + 'lower.npy', lowers)
        np.save(folder_path + 'upper.npy', uppers)

    @staticmethod
    def _winkler_score(y, lower, upper, alpha):
        width = upper - lower
        below = y < lower
        above = y > upper
        score = width.copy()
        score[below] += 2.0 / alpha * (lower[below] - y[below])
        score[above] += 2.0 / alpha * (y[above] - upper[above])
        return np.mean(score)
