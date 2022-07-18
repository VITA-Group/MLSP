import datetime
import pandas as pd
import copy
import torch
import os
import sklearn.metrics as metrics
from PointDA.data.dataloader import label_to_idx
import logging

class IOStream():
    """
    Logging to screen and file
    """
    def __init__(self, args):
        self.path = args.out_path + '/' + args.exp_name
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.f = open(self.path + '/run.log', 'a')
        self.args = args

    def cprint(self, text):
        datetime_string = datetime.datetime.now().strftime("%d-%m-%y %H:%M:%S")
        to_print = "%s: %s" % (datetime_string, text)
        print(to_print)
        self.f.write(to_print + "\n")
        self.f.flush()

    def close(self):
        self.f.close()

    def save_model(self, model):
        path = self.path + '/model.pt'
        best_model = copy.deepcopy(model)
        torch.save(model.state_dict(), path)
        return best_model

    def save_best_model(self, model):
        path = self.path + '/best_model.pt'
        best_model = copy.deepcopy(model)
        torch.save(model.state_dict(), path)
        return best_model

    def save_conf_mat(self, conf_matrix, fname, domain_set):
        df = pd.DataFrame(conf_matrix, columns=list(label_to_idx.keys()), index=list(label_to_idx.keys()))
        fname = domain_set + "_" + fname
        df.to_csv(self.path + "/" + fname)

    def print_progress(self, domain_set, partition, epoch, print_losses, true=None, pred=None):
        outstr = "%s - %s %d" % (partition, domain_set, epoch)
        acc = 0
        if true is not None and pred is not None:
            acc = metrics.accuracy_score(true, pred)
            avg_per_class_acc = metrics.balanced_accuracy_score(true, pred)
            outstr += ", acc: %.4f, avg acc: %.4f" % (acc, avg_per_class_acc)
        if print_losses is not None:
            for loss, loss_val in print_losses.items():
                outstr += ", %s loss: %.4f" % (loss, loss_val)
        self.cprint(outstr)
        return acc

# def print_log(msg, logger=None, level=logging.INFO):
#     """Print a log message.
#     Args:
#         msg (str): The message to be logged.
#         logger (logging.Logger | str | None): The logger to be used.
#             Some special loggers are:
#             - "silent": no message will be printed.
#             - other str: the logger obtained with `get_root_logger(logger)`.
#             - None: The `print()` method will be used to print log messages.
#         level (int): Logging level. Only available when `logger` is a Logger
#             object or "root".
#     """
#     if logger is None:
#         print(msg)
#     elif isinstance(logger, logging.Logger):
#         logger.log(level, msg)
#     elif logger == 'silent':
#         pass
#     elif isinstance(logger, str):
#         _logger = get_logger(logger)
#         _logger.log(level, msg)
#     else:
#         raise TypeError(
#             'logger should be either a logging.Logger object, str, '
#             f'"silent" or None, but got {type(logger)}')