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
from torch.utils.data import DataLoader, Subset

import os
import time


class Exp_APEC(Exp_Basic):
    def __init__(self, args):
        super(Exp_APEC, self).__init__(args)
        self.plugin = None
        self.q = None
        self.gamma = 1.0

    def _build_plugin(self):
        if self.plugin is not None:
            return self.plugin
        n_channels = 1 if self.args.features == 'MS' else self.args.enc_in
        self.plugin = APEC.TrendPlugin(
            pred_len=self.args.pred_len,
            n_channels=n_channels,
        ).to(self.device)
        return self.plugin

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

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _target_offset(self):
        return -1 if self.args.features == 'MS' else 0

    def _target_slice(self, batch_y):
        return batch_y[:, -self.args.pred_len:, self._target_offset():]

    def _make_loader(self, dataset, shuffle=False):
        return DataLoader(dataset, batch_size=self.args.batch_size, shuffle=shuffle,
                          num_workers=self.args.num_workers, drop_last=False)

    def _forward_backbone(self, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle):
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self._dispatch_backbone(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle)
        else:
            outputs = self._dispatch_backbone(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle)
        return outputs[:, -self.args.pred_len:, self._target_offset():]

    def _dispatch_backbone(self, batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle):
        if any(s in self.args.model for s in ('CycleNet', 'GTR')):
            return self.model(batch_x, batch_cycle)
        if any(s in self.args.model for s in ('Linear', 'MLP', 'SegRNN', 'TST')):
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
                total_loss.append(criterion(outputs.detach().cpu(), true.detach().cpu()).item())
        self.model.train()
        return np.average(total_loss)

    def _split_posthoc_indices(self, data_len):
        start = min(self.args.apec_window, max(0, data_len - 3))
        usable = data_len - start
        plugin_end = start + max(1, int(usable * self.args.apec_val_plugin_ratio))
        gamma_end = plugin_end + max(1, int(usable * self.args.apec_val_gamma_ratio))
        plugin_end = min(plugin_end, data_len - 2)
        gamma_end = min(max(gamma_end, plugin_end + 1), data_len - 1)
        return range(start, plugin_end), range(plugin_end, gamma_end), range(gamma_end, data_len)

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

    # ── plugin helpers ────────────────────────────────────────────────────────

    def _delta(self, batch_size):
        return self.plugin.T.unsqueeze(0).expand(batch_size, -1, -1)

    def _apply_gamma(self, delta, gamma=None):
        if gamma is None:
            gamma = self.gamma
        if isinstance(gamma, torch.Tensor):
            g = gamma.to(device=delta.device, dtype=delta.dtype).view(1, -1, 1)
        else:
            g = torch.tensor(gamma, device=delta.device, dtype=delta.dtype)
        return g * delta

    def _gamma_for_save(self):
        return float(self.gamma) if not isinstance(self.gamma, torch.Tensor) else self.gamma.detach().cpu()

    def _gamma_summary(self):
        return "{:.2f}".format(float(self.gamma))

    def _gamma_metric_value(self):
        return float(self.gamma)

    def _eval_plugin_mse(self, loader, gamma=1.0):
        corrected_losses, base_losses = [], []
        self.model.eval()
        self.plugin.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle in loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)
                y_hat = self._forward_backbone(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle)
                true = self._target_slice(batch_y)
                pred = y_hat + gamma * self._delta(y_hat.shape[0])
                corrected_losses.append(torch.mean((true - pred) ** 2).item())
                base_losses.append(torch.mean((true - y_hat) ** 2).item())
        return np.average(corrected_losses), np.average(base_losses)

    def _select_gamma(self, eval_loader):
        if self.args.apec_gamma_step <= 0:
            self.gamma = 1.0
            return self.gamma
        gamma_values = np.arange(0.0, 1.0 + 0.5 * self.args.apec_gamma_step, self.args.apec_gamma_step)
        best_gamma, best_mse, baseline_mse = 0.0, None, None
        print("APEC gamma sweep:")
        for gamma in gamma_values:
            mse, base_mse = self._eval_plugin_mse(eval_loader, gamma=float(gamma))
            if baseline_mse is None:
                baseline_mse = base_mse
            marker = ""
            if best_mse is None or mse < best_mse:
                best_mse = mse
                best_gamma = float(gamma)
                marker = " <-- best"
            print("  gamma={:.2f}  corrected={:.6f}  base={:.6f}  delta={:+.6f}{}".format(
                gamma, mse, base_mse, mse - base_mse, marker))
        threshold = getattr(self.args, 'apec_gamma_min_improve', 0.01)
        if baseline_mse is not None and best_mse is not None:
            if (baseline_mse - best_mse) < threshold * max(baseline_mse, 1e-8):
                best_gamma = 0.0
                print("  improvement {:.4f}% < threshold {:.1f}%, forcing gamma=0.0".format(
                    100.0 * (baseline_mse - best_mse) / max(baseline_mse, 1e-8), threshold * 100))
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
            train_loss = []
            epoch_time = time.time()

            for batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle in plugin_loader:
                optimizer.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)
                with torch.no_grad():
                    y_hat = self._forward_backbone(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle)
                true = self._target_slice(batch_y)
                pred = y_hat + self._delta(y_hat.shape[0])
                loss = torch.mean((true - pred) ** 2)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())

            print("APEC Epoch: {0} cost time: {1:.3f}".format(epoch + 1, time.time() - epoch_time))
            val_mse, base_val_mse = self._eval_plugin_mse(plugin_val_loader, gamma=1.0)
            print("APEC Epoch: {0} | Train Loss: {1:.7f} | Val MSE: {2:.7f} Base Val MSE: {3:.7f}".format(
                epoch + 1, np.average(train_loss), val_mse, base_val_mse))

            if best_val_mse is None or val_mse < best_val_mse:
                best_val_mse = val_mse
                bad_epochs = 0
                torch.save(self.plugin.state_dict(), best_path)
            else:
                bad_epochs += 1
                print("APEC early stopping counter: {} out of {}".format(bad_epochs, self.args.apec_plugin_patience))
                if bad_epochs >= self.args.apec_plugin_patience:
                    print("APEC plug-in early stopping")
                    break

        self.plugin.load_state_dict(torch.load(best_path))
        gamma = self._select_gamma(plugin_val_loader)
        torch.save({'gamma': self._gamma_for_save()}, os.path.join(path, 'apec_gamma.pt'))
        val_mse, base_val_mse = self._eval_plugin_mse(plugin_val_loader, gamma=gamma)
        print("APEC selected gamma: {} | Shrunk Val MSE: {:.7f} Base Val MSE: {:.7f}".format(
            self._gamma_summary(), val_mse, base_val_mse))

    def _calibrate(self, cal_loader, path):
        scores = []
        self.model.eval()
        self.plugin.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle in cal_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)
                y_hat = self._forward_backbone(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle)
                true = self._target_slice(batch_y)
                pred = y_hat + self._apply_gamma(self._delta(y_hat.shape[0]))
                scores.append(torch.abs(true - pred).detach().cpu())
        scores = torch.cat(scores, dim=0)                          # [N, H, C]
        self.q = torch.quantile(scores, 1 - self.args.apec_alpha, dim=0)  # [H, C]
        torch.save({'q': self.q, 'alpha': self.args.apec_alpha, 'gamma': self._gamma_for_save()},
                   os.path.join(path, 'apec_q.pt'))
        print("APEC conformal q calibrated with shape {}".format(tuple(self.q.shape)))

    # ── train / test ──────────────────────────────────────────────────────────

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        plugin_idx, cal_idx, plugin_val_idx = self._split_posthoc_indices(len(vali_data))
        print("APEC split sizes | backbone_train: {} backbone_val: {} plugin: {} plugin_val: {} calibration: {}".format(
            len(train_data), len(vali_data), len(plugin_idx), len(plugin_val_idx), len(cal_idx)))

        print(">>>>>>>stage 1: train frozen backbone : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting))
        self._train_backbone(setting, train_loader, vali_loader, path)

        for param in self.model.parameters():
            param.requires_grad = False
        self._build_plugin()

        plugin_loader     = self._make_loader(Subset(vali_data, list(plugin_idx)),     shuffle=True)
        plugin_val_loader = self._make_loader(Subset(vali_data, list(plugin_val_idx)), shuffle=False)
        cal_loader        = self._make_loader(Subset(vali_data, list(cal_idx)),        shuffle=False)

        print(">>>>>>>stage 2: train APEC plug-in<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        self._train_plugin(plugin_loader, plugin_val_loader, path)
        print(">>>>>>>stage 3: conformal calibration<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        self._calibrate(cal_loader, path)
        return self.model

    def test(self, setting, test=0):
        _, test_loader = self._get_data(flag='test')
        path = os.path.join(self.args.checkpoints, setting)
        self._build_plugin()

        if test:
            print('loading APEC backbone, plug-in, and q')
            self.model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth')))
            self.plugin.load_state_dict(torch.load(os.path.join(path, 'apec_plugin.pth')))
        if self.q is None:
            q_path = os.path.join(path, 'apec_q.pt')
            if not os.path.exists(q_path):
                raise RuntimeError('APEC q is not calibrated. Run training before test.')
            q_state = torch.load(q_path)
            self.q = q_state['q']
            self.gamma = q_state.get('gamma', torch.load(os.path.join(path, 'apec_gamma.pt'))['gamma'])
        print("APEC test gamma: {}".format(self._gamma_summary()))

        preds, base_preds, trues, lowers, uppers = [], [], [], [], []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        q = self.q.to(self.device)
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
                pred = y_hat + self._apply_gamma(self._delta(y_hat.shape[0]))
                lower = pred - q
                upper = pred + q
                preds.append(pred.detach().cpu().numpy())
                base_preds.append(y_hat.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())
                lowers.append(lower.detach().cpu().numpy())
                uppers.append(upper.detach().cpu().numpy())
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
        lowers     = np.concatenate(lowers,     axis=0)
        uppers     = np.concatenate(uppers,     axis=0)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        base_mae, base_mse, *_ = metric(base_preds, trues)
        coverage = np.mean((trues >= lowers) & (trues <= uppers))
        mean_width = np.mean(uppers - lowers)
        winkler = self._winkler_score(trues, lowers, uppers, self.args.apec_alpha)

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print('base_mse:{}, base_mae:{}'.format(base_mse, base_mae))
        print('apec_mse:{}, apec_mae:{}, coverage:{}, width:{}, winkler:{}, gamma:{}'.format(
            mse, mae, coverage, mean_width, winkler, self._gamma_summary()))
        with open("result.txt", 'a') as f:
            f.write(setting + "  \n")
            f.write('base_mse:{}, base_mae:{}\n'.format(base_mse, base_mae))
            f.write('apec_mse:{}, apec_mae:{}, coverage:{}, width:{}, winkler:{}, gamma:{}\n\n'.format(
                mse, mae, coverage, mean_width, winkler, self._gamma_summary()))

        np.save(folder_path + 'metrics_apec.npy', np.array([mae, mse, coverage, mean_width, winkler, self._gamma_metric_value()]))
        np.save(folder_path + 'pred_apec.npy', preds)
        np.save(folder_path + 'pred_base.npy', base_preds)
        np.save(folder_path + 'true.npy', trues)
        np.save(folder_path + 'lower.npy', lowers)
        np.save(folder_path + 'upper.npy', uppers)

    @staticmethod
    def _winkler_score(y, lower, upper, alpha):
        width = upper - lower
        score = width.copy()
        score[y < lower] += 2.0 / alpha * (lower[y < lower] - y[y < lower])
        score[y > upper] += 2.0 / alpha * (y[y > upper] - upper[y > upper])
        return np.mean(score)
