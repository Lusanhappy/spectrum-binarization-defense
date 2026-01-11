import os, datetime, time
import torch, random
import utils
from tqdm import tqdm
from typing import Tuple
from utils import get_logger
from dataset import AdvDataset
from models import load_model
from collections import Counter
from argparse import ArgumentParser
from default_config import model_config as mcfg
from default_config import train_config as tcfg
from torchsummary import summary

logger = get_logger(name='train')

DEBUG = tcfg.debug_info

parser = ArgumentParser()
parser.add_argument('--data_path', type=str, default=None, help='Traning dataset path.')
parser.add_argument('--restore_from', type=str, default=None, help='Initialize the model with specified weight.')
parser.add_argument('--epoch', type=int, default=None, help='Training epochs.')
parser.add_argument('--lr', type=float, default=None, help='Learning rate.')
parser.add_argument('--weight_decay', type=float, default=None, help='Weight decay.')
parser.add_argument('--save_path', type=str, default=None, help='Model weights and logs save path.')
parser.add_argument('--batch', type=int, default=None, help='Batch size.')
parser.add_argument('--save_interval', type=int, default=None, help='Save the model weights every n epochs.')
parser.add_argument('--random_noise_amp', type=float, default=None, help='Random noise amplitude (0-1) in data augmentation.')
parser.add_argument('--model_name', type=str, default=None, help='cnn_attention/cnn3_dnn2')
parser.add_argument('--dual_feats', type=int, default=None, help='Use both the original feature and its 1st difference .')
parser.add_argument('--use_binary_features', type=int, default=None, help='Use binary linear spectrogram as feature.')
parser.add_argument('--use_melspec', type=int, default=None, help='Use mel spectrogram as feature.')
parser.add_argument('--melspec_diff', type=int, default=None, help="Use mel spectrogram's 1st difference.")
parser.add_argument('--use_formants', type=int, default=None, help='Use formants as feature.')
parser.add_argument('--formants_diff', type=int, default=None, help="Use formants' 1st difference.")
parser.add_argument('--use_pitch', type=int, default=None, help='Use pitch as feature.')
parser.add_argument('--pitch_diff', type=int, default=None, help="Use pitch's 1st difference.")

args = parser.parse_args()
for k, v in args.__dict__.items():
    if v is not None:
        if k in tcfg.__dict__.keys():
            tcfg.__dict__[k] = type(tcfg.__dict__[k])(v)
            logger.info(f'Train config set: {k}={tcfg.__dict__[k]}')
        elif k in mcfg.__dict__.keys():
            mcfg.__dict__[k] = type(mcfg.__dict__[k])(v)
            logger.info(f'Model config set: {k}={mcfg.__dict__[k]}')
        else:
            logger.info(f'arg: {k} cannot be parsed.')

######################################################################################

def _count_top1_per_class(outputs: torch.Tensor,
                          targets: torch.Tensor) -> Tuple[Counter, Counter]:
    pred_labels = torch.argmax(outputs, dim=-1).view(-1)
    true_labels = targets.view(-1)
    true_or_false = torch.eq(pred_labels, true_labels)
    top1 = Counter(torch.masked_select(true_labels, true_or_false).tolist())
    total = Counter(true_labels.tolist())
    return top1, total


time_str = lambda: datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


class Trainer():
    def __init__(self):
        if DEBUG is True:
            logger.debug(f'\nTrainer parameters received: {str(tcfg)}\n')
            logger.debug(f'\nTrainer parameters received:')
            for k, v in mcfg.__dict__.items():
                logger.debug(f'{k}: {str(v)}')
        self._init_task()

    def _init_task(self):
        random.seed(tcfg.random_seed)
        # Get dataset
        self.dataset = AdvDataset()
        if DEBUG is True:
            logger.debug(f'Dataset: {str(self.dataset)}')
        # Get model
        self.model = load_model(mcfg.model_name).cuda()
        # summary(self.model, input_size=(1, 16000))
        if tcfg.restore_from is not None:
            self.model.load_state_dict(torch.load(tcfg.restore_from))
            logger.info(f'Restore from: {tcfg.restore_from}')
        # Log parameters
        tcfg.save_path = os.path.join(tcfg.save_path, time_str())
        os.makedirs(tcfg.save_path)
        utils.write_json(
            os.path.join(tcfg.save_path, 'config_train.json'), 
            tcfg.__dict__)
        utils.write_json(
            os.path.join(tcfg.save_path, 'config_model.json'), 
            mcfg.__dict__)
        # Get training environment
        self.named_params: dict = dict(
            filter(lambda kv: kv[1].requires_grad,
                   self.model.named_parameters()))
        if DEBUG is True:
            logger.debug(f'Optimizable params: {str(list(self.named_params.keys()))}')
        self.optimizer = torch.optim.Adam(self.named_params.values(),
                                          tcfg.lr,
                                          weight_decay=tcfg.weight_decay)
        self.criterion = torch.nn.CrossEntropyLoss().cuda()
        if tcfg.no_val is False:
            self.train_loader, self.val_loader = \
                self.dataset.get_loaders(tcfg.batch)
        else:
            self.train_loader = self.dataset.get_loaders(tcfg.batch)

    def _save_model(self, *args: list):
        save_path = os.path.join(
            tcfg.save_path, '_'.join(args) + '.pth')
        torch.save(self.model.state_dict(), save_path)
        if DEBUG is True:
            logger.debug(f'Model saved to: {save_path}')
        return save_path

    def _forward(self, 
        waves: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        if mcfg.model_name == 'cnn3_dnn2':
            waves = waves.cuda(non_blocking=True)
            feats = feats.cuda(non_blocking=True)
            outputs = self.model(waves, feats)
        elif mcfg.model_name == 'cnn_attention':
            waves = waves.cuda(non_blocking=True)
            outputs = self.model(waves)
        else:
            raise ValueError(f'Invalid config set "model_name"={mcfg.model_name}')
        return outputs

    def validate(self) -> float:
        """Validate for one epoch
        """
        n_steps = len(self.val_loader)
        progress = tqdm(self.val_loader)
        top1, total = Counter(), Counter()
        with torch.no_grad():
            for i_step, (waves, feats, targets) in enumerate(progress):
                targets = targets.cuda(non_blocking=True)
                outputs = self._forward(waves, feats)
                # Compute Metric
                top1_batch, total_batch = _count_top1_per_class(
                    outputs, targets)
                top1.update(top1_batch)
                total.update(total_batch)
                progress.set_description(
                    'Validation: Step {0}/{1} ACC: {2:.4f}'.format(
                        i_step, n_steps,
                        sum(top1.values()) / sum(total.values())))
        return sum(top1.values()) / sum(total.values())

    def train_one_epoch(self) -> float:
        """Train for one epoch
        """
        n_steps = len(self.train_loader)
        progress = tqdm(self.train_loader)
        top1, total = Counter(), Counter()
        self.optimizer.zero_grad()
        for i_step, (waves, feats, targets) in enumerate(progress):
            targets = targets.cuda(non_blocking=True)
            outputs = self._forward(waves, feats)
            loss = self.criterion(outputs, targets)
            # Backward propagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            top1_batch, total_batch = _count_top1_per_class(outputs, targets)
            top1.update(top1_batch)
            total.update(total_batch)
            progress.set_description(
                'Train: Step: {0}/{1} loss {2:.4f} ACC: {3:.4f}'.format(
                    i_step, n_steps, loss.item(),
                    sum(top1.values()) / sum(total.values())))
        return sum(top1.values()) / sum(total.values())

    def train_loop(self):
        # Epoch loop: Train and Validate
        with open(os.path.join(tcfg.save_path, 'log.txt'), 'w') as logf:
            start_time = time.time()
            # Validate initial val acc
            if tcfg.no_val is False:
                self.model.eval()
                val_acc_best = self.validate()
                logf.write(' '.join([time_str(), f'val-acc:{val_acc_best:.4f}', '\n']))
            # Train loop
            for i_epoch in range(tcfg.epoch):
                logger.debug(f'Training epoch {i_epoch + 1}')
                # Train one epoch
                self.model.train()
                train_acc = self.train_one_epoch()
                logf.write(' '.join([time_str(), f'train-acc:{train_acc:.4f}', '\n']))
                # Validation after one epoch
                if tcfg.no_val is False:
                    self.model.eval()
                    val_acc = self.validate()
                    logf.write(' '.join([time_str(), f'val-acc:{val_acc:.4f}', '\n']))
                # Save model weights optionally
                if (i_epoch + 1) % tcfg.save_interval == 0 \
                     or (tcfg.no_val is False and val_acc > val_acc_best):
                    if tcfg.no_val is False:
                        if val_acc > tcfg.save_valacc_bound:
                            # save_path = \
                            self._save_model(
                                f'epoch-{i_epoch + 1}', time_str(), 
                                f'trnacc-{train_acc:.4f}', f'valacc-{val_acc:.4f}')
                            # self.model.load_state_dict(torch.load(save_path))
                        val_acc_best = max(val_acc, val_acc_best)
                    else:
                        self._save_model(
                            f'epoch-{i_epoch + 1}', 
                            time_str(), 
                            f'trnacc-{train_acc:.4f}')
            # Save model weights
            if tcfg.no_val is False:
                save_path = self._save_model(
                    f'epoch-{i_epoch + 1}', time_str(), f'trnacc-{train_acc:.4f}', f'valacc-{val_acc:.4f}')
            else:
                save_path = self._save_model(
                    f'epoch-{i_epoch + 1}', time_str(), f'trnacc-{train_acc:.4f}')
            lapsed = time.time() - start_time
            logger.info(f'{lapsed / 60:.4f} min ({lapsed / 60 / 60:.4f} h) cost.')
            return save_path

if __name__ == '__main__':
    trainer = Trainer()
    save_path = trainer.train_loop()
    # trainer = Trainer()
    trainer.model.load_state_dict(torch.load(save_path))
    trainer.model.eval()
    if tcfg.no_val is False:
        trainer.validate()
