# -*- coding: UTF-8 -*-

import os
import gc
import torch
import torch.nn as nn
import logging
import numpy as np
from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, List
import pickle

from utils import utils
from utils.constants import *
from utils.metrics import *
from models.BaseModel import BaseModel


class BaseRunner(object):
    @staticmethod
    def parse_runner_args(parser):
        parser.add_argument("--epoch", type=int, default=300, help="Number of epochs.")
        parser.add_argument(
            "--check_epoch",
            type=int,
            default=0,
            help="Check some tensors every check_epoch.",
        )
        parser.add_argument(
            "--test_epoch",
            type=int,
            default=-1,
            help="Print test results every test_epoch (-1 means no print).",
        )
        parser.add_argument(
            "--early_stop",
            type=int,
            default=10,
            help="The number of epochs when valid results drop continuously.",
        )
        parser.add_argument(
            "--min_best_epoch",
            type=int,
            default=1,
            help=(
                "Earliest 1-based epoch eligible for best-valid checkpoint "
                "selection and early-stop tracking. Default 1 preserves the "
                "standard protocol."
            ),
        )
        parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
        parser.add_argument(
            "--l2", type=float, default=0, help="Weight decay in optimizer."
        )
        parser.add_argument(
            "--batch_size", type=int, default=256, help="Batch size during training."
        )
        parser.add_argument(
            "--eval_batch_size",
            type=int,
            default=1024,
            help="Batch size during testing.",
        )
        parser.add_argument(
            "--optimizer",
            type=str,
            default="Adam",
            help="optimizer: SGD, Adam, Adagrad, Adadelta",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=10,
            help="Number of processors when prepare batches in DataLoader",
        )
        parser.add_argument(
            "--pin_memory", type=int, default=0, help="pin_memory in DataLoader"
        )
        parser.add_argument(
            "--topk",
            type=str,
            default="5,10,20,50,100",
            help="The number of items recommended to each user.",
        )
        parser.add_argument(
            "--metric",
            type=str,
            default="NDCG,HR,MRR",
            help="metrics: NDCG, HR, MRR, PRECISION, MAP",
        )
        parser.add_argument(
            "--main_metric",
            type=str,
            default="NDCG@20",
            help="Main metric to determine the best model.",
        )
        parser.add_argument(
            "--clip_grad",
            action="store_true",
            help="Whether to use gradient clip or not.",
        )
        parser.add_argument(
            "--max_norm",
            type=float,
            default=1.0,
            help="Max norm of the gradients for gradient clipping",
        )
        parser.add_argument(
            "--warmup_steps",
            type=int,
            default=0,
            help="Number of warmup steps for learning rate warmup (0 means no warmup)",
        )
        parser.add_argument(
            "--warmup_epochs",
            type=int,
            default=0,
            help="Number of warmup epochs for learning rate warmup (0 means no warmup). "
            "If both warmup_steps and warmup_epochs are set, warmup_steps takes priority.",
        )
        parser.add_argument(
            "--lr_scheduler",
            type=str,
            default="constant",
            choices=["constant", "cosine", "linear_decay"],
            help="Learning rate scheduler after warmup: constant (no decay), "
            "cosine (cosine annealing), linear_decay (linear decay to 0)",
        )
        return parser

    @staticmethod
    def _topk_indices_from_predictions(
        predictions, topk: list
    ) -> np.ndarray:
        """Compute top-k indices from a score matrix without keeping extra GPU refs."""
        if not torch.is_tensor(predictions):
            predictions = torch.from_numpy(predictions)

        max_topk = max(topk)
        batch_size = 65536
        top_indices_list = []
        for start in range(0, predictions.shape[0], batch_size):
            end = min(start + batch_size, predictions.shape[0])
            batch = predictions[start:end]
            _, topk_idx = torch.topk(batch, k=max_topk, dim=1)
            top_indices_list.append(topk_idx.cpu().numpy())
            del batch, topk_idx
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return np.vstack(top_indices_list)

    @staticmethod
    def evaluate_method_from_top_indices(
        top_idx, targets, topk: list, metrics: list
    ) -> Dict[str, float]:
        """Evaluate ranking metrics from precomputed top-k indices."""
        if torch.is_tensor(top_idx):
            top_idx = top_idx.cpu().numpy()
        if torch.is_tensor(targets):
            targets = targets.cpu().numpy()

        evaluations = dict()
        pos_matrix = top_idx == targets.reshape(-1, 1)
        for m in metrics:
            m_res = eval(m)(pos_matrix, topk)
            for k, res in zip(topk, m_res):
                evaluations[m + "@" + str(k)] = round(res.mean(), 4)

        return evaluations

    @staticmethod
    def evaluate_method(
        predictions: np.ndarray, targets: np.ndarray, topk: list, metrics: list
    ) -> Dict[str, float]:
        """
        :param predictions: (-1, n_candidates) shape, the first column is the score for ground-truth item
        :param targets: (-1,) shape, the ground-truth item id
        :param topk: top-K value list
        :param metrics: metric string list
        :return: a result dict, the keys are metric@k
        """
        top_idx = BaseRunner._topk_indices_from_predictions(predictions, topk)
        return BaseRunner.evaluate_method_from_top_indices(
            top_idx, targets, topk, metrics
        )

    def __init__(self, args):
        self.train_models = args.train
        self.epoch = args.epoch
        self.check_epoch = args.check_epoch
        self.test_epoch = args.test_epoch
        self.early_stop = args.early_stop
        self.min_best_epoch = max(1, int(getattr(args, "min_best_epoch", 1)))
        self.learning_rate = args.lr
        self.batch_size = args.batch_size
        self.eval_batch_size = args.eval_batch_size
        self.l2 = args.l2
        self.optimizer_name = args.optimizer
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.topk = [int(x) for x in args.topk.split(",")]
        self.metrics = [m.strip().upper() for m in args.metric.split(",")]
        self.main_metric = args.main_metric  # early stop based on main_metric
        self.main_topk = int(self.main_metric.split("@")[1])
        self.clip_grad = args.clip_grad
        self.max_norm = args.max_norm
        self.device = args.device

        # Learning rate warmup and scheduler settings
        self.warmup_steps = args.warmup_steps
        self.warmup_epochs = args.warmup_epochs
        self.lr_scheduler_type = args.lr_scheduler
        self.current_step = 0  # Track global training step for warmup
        self.total_steps = None  # Will be set during training
        self.best_epoch = None
        self.best_valid_result = None
        self.train_wall_time = None

        self.time = None  # will store [start_time, last_step_time]

        self.log_path = os.path.dirname(args.log_file)  # path to save predictions
        self.save_appendix = args.log_file.split("/")[-1].split(".")[
            0
        ]  # appendix for prediction saving
        
        self.is_main_process = True

    def _check_time(self, start=False):
        if self.time is None or start:
            self.time = [time()] * 2
            return self.time[0]
        tmp_time = self.time[1]
        self.time[1] = time()
        return self.time[1] - tmp_time

    def _build_optimizer(self, model):
        logging.info("Optimizer: " + self.optimizer_name)
        optimizer = eval("torch.optim.{}".format(self.optimizer_name))(
            model.customize_parameters(), lr=self.learning_rate, weight_decay=self.l2
        )
        return optimizer

    def _get_lr_multiplier(self):
        """
        Calculate learning rate multiplier based on current step and warmup settings.
        
        Returns:
            float: multiplier for the base learning rate (1.0 = full learning rate)
        """
        # No warmup
        if self.warmup_steps == 0 or self.current_step >= self.warmup_steps:
            # Apply learning rate decay after warmup if specified
            if self.lr_scheduler_type == "constant":
                return 1.0
            elif self.lr_scheduler_type == "cosine":
                # Cosine annealing after warmup
                if self.total_steps is None or self.total_steps <= self.warmup_steps:
                    return 1.0
                progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                return 0.5 * (1.0 + np.cos(np.pi * progress))
            elif self.lr_scheduler_type == "linear_decay":
                # Linear decay to 0 after warmup
                if self.total_steps is None or self.total_steps <= self.warmup_steps:
                    return 1.0
                progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                return max(0.0, 1.0 - progress)
            else:
                return 1.0
        else:
            # Linear warmup: gradually increase from 0 to 1
            return self.current_step / self.warmup_steps
    
    def _update_learning_rate(self, optimizer):
        """
        Update learning rate for all parameter groups in optimizer.
        """
        lr_multiplier = self._get_lr_multiplier()
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.learning_rate * lr_multiplier

    def train(self, data_dict: Dict[str, BaseModel.Dataset]):
        model = data_dict["train"].model
        main_metric_results, valid_results = list(), list()
        self._check_time(start=True)
        
        # Calculate total steps for learning rate scheduling.
        effective_dataset_size = len(data_dict["train"])
        steps_per_epoch = effective_dataset_size // self.batch_size + (
            1 if effective_dataset_size % self.batch_size != 0 else 0
        )
        self.total_steps = self.epoch * steps_per_epoch
        
        # Set warmup steps from epochs if specified
        if self.warmup_epochs > 0 and self.warmup_steps == 0:
            self.warmup_steps = self.warmup_epochs * steps_per_epoch
        
        # LoopRec warmup epoch-to-step conversion (if the model supports it)
        if hasattr(model, 'set_looprec_warmup_steps'):
            model.set_looprec_warmup_steps(steps_per_epoch)

        # Log warmup configuration (only on main process)
        if self.warmup_steps > 0 and self.is_main_process:
            logging.info(f"Learning rate warmup enabled:")
            logging.info(f"  - Warmup steps: {self.warmup_steps}")
            logging.info(f"  - Warmup epochs: ~{self.warmup_steps / steps_per_epoch:.2f}")
            logging.info(f"  - LR scheduler after warmup: {self.lr_scheduler_type}")
            logging.info(f"  - Total training steps: {self.total_steps}")

        should_stop = False
        model_saved = False
        
        try:
            for epoch in range(self.epoch):
                # Fit
                self._check_time()
                gc.collect()
                torch.cuda.empty_cache()
                loss = self.fit(data_dict["train"], epoch=epoch + 1)
                if np.isnan(loss):
                    if self.is_main_process:
                        logging.info("Loss is Nan. Stop training at %d." % (epoch + 1))
                    should_stop = True
                
                if not should_stop:
                    training_time = self._check_time()

                    # Observe selected tensors (only on main process)
                    if (
                        self.is_main_process
                        and len(model.check_list) > 0
                        and self.check_epoch > 0
                        and epoch % self.check_epoch == 0
                    ):
                        utils.check(model.check_list)

                    # Record valid results (only on main process)
                    if self.is_main_process:
                        valid_result = self.evaluate(
                            data_dict["valid"], [self.main_topk], self.metrics
                        )
                        valid_results.append(valid_result)
                        main_metric_results.append(valid_result[self.main_metric])
                        logging_str = "Epoch {:<5} loss={:<.4f} [{:<3.1f} s]	valid=({})".format(
                            epoch + 1, loss, training_time, utils.format_metric(valid_result)
                        )

                        # Test
                        if self.test_epoch > 0 and epoch % self.test_epoch == 0:
                            test_result = self.evaluate(
                                data_dict["test"], self.topk[:1], self.metrics
                            )
                            logging_str += " test=({})".format(utils.format_metric(test_result))
                        
                        testing_time = self._check_time()
                        logging_str += " [{:<.1f} s]".format(testing_time)

                        # Select checkpoints and track early stopping from min_best_epoch.
                        epoch_num = epoch + 1
                        selectable_metric_results = (
                            main_metric_results[self.min_best_epoch - 1 :]
                            if epoch_num >= self.min_best_epoch
                            else []
                        )
                        is_selectable_best = (
                            len(selectable_metric_results) > 0
                            and max(selectable_metric_results) == selectable_metric_results[-1]
                        )
                        if is_selectable_best or (
                            epoch_num >= self.min_best_epoch
                            and hasattr(model, "stage")
                            and model.stage == 1
                        ):
                            model.save_model()
                            model_saved = True
                            logging_str += " *"
                        logging.info(logging_str)

                        early_stop_metric_results = (
                            main_metric_results[self.min_best_epoch - 1 :]
                            if len(main_metric_results) >= self.min_best_epoch
                            else []
                        )
                        if (
                            self.early_stop > 0
                            and len(early_stop_metric_results) > 0
                            and self.eval_termination(early_stop_metric_results)
                        ):
                            logging.info(
                                "Early stop at %d based on valid result." % (epoch + 1)
                            )
                            should_stop = True
                
                if should_stop:
                    break

        except KeyboardInterrupt:
            if self.is_main_process:
                logging.info("Early stop manually")
                exit_here = input("Exit completely without evaluation? (y/n) (default n):")
                if exit_here.lower().startswith("y"):
                    logging.info(
                        os.linesep + "-" * 45 + " END: " + utils.get_time() + " " + "-" * 45
                    )
                    exit(1)

        # Find the best valid result across iterations
        if self.is_main_process and len(main_metric_results) > 0:
            if len(main_metric_results) >= self.min_best_epoch:
                eligible_metric_results = main_metric_results[self.min_best_epoch - 1 :]
                best_epoch = (
                    self.min_best_epoch
                    - 1
                    + eligible_metric_results.index(max(eligible_metric_results))
                )
            else:
                logging.warning(
                    "No validation epoch reached min_best_epoch=%d; falling back to "
                    "global best-valid reporting.",
                    self.min_best_epoch,
                )
                best_epoch = main_metric_results.index(max(main_metric_results))
            self.best_epoch = best_epoch + 1
            self.best_valid_result = valid_results[best_epoch]
            self.train_wall_time = self.time[1] - self.time[0]
            if self.min_best_epoch > 1:
                logging.info("Best-valid selection min_best_epoch=%d", self.min_best_epoch)
            logging.info(
                os.linesep
                + "Best Iter(valid)={:>5}\t valid=({}) [{:<.1f} s] ".format(
                    best_epoch + 1,
                    utils.format_metric(valid_results[best_epoch]),
                    self.time[1] - self.time[0],
                )
            )
        
        if model_saved:
            model.load_model()
        else:
            if self.is_main_process:
                logging.warning("No model was saved during training (possibly due to early NaN). Skipping model load.")

    def fit(self, dataset: BaseModel.Dataset, epoch=-1) -> float:
        model = dataset.model

        if model.optimizer is None:
            model.optimizer = self._build_optimizer(model)
        
        dataset.actions_before_epoch()  # must sample before multi thread start

        model.train()
        
        loss_lst = list()
        
        # Note: persistent_workers and pin_memory together cause memory/file handle leaks
        # in PyTorch. Only enable persistent_workers when pin_memory is disabled.
        use_persistent = (self.num_workers > 0) and (not self.pin_memory)
        
        dl = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=dataset.collate_batch,
            pin_memory=self.pin_memory,
            persistent_workers=use_persistent,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )
        
        # Only show progress bar on main process
        if self.is_main_process:
            dl_iter = tqdm(dl, leave=False, desc="Epoch {:<3}".format(epoch), ncols=100, mininterval=1)
        else:
            dl_iter = dl
        
        for batch in dl_iter:
            batch = utils.batch_to_gpu(batch, model.device)

            # Increment step counter BEFORE updating learning rate
            # This ensures the first batch gets lr = base_lr * (1/warmup_steps) instead of 0
            self.current_step += 1
            
            self._update_learning_rate(model.optimizer)

            model.optimizer.zero_grad()
            out_dict = model(batch, epoch=epoch)

            loss = model.loss(out_dict)
            loss.backward()
            if self.clip_grad:
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    self.max_norm,
                )
            model.optimizer.step()

            # Optional model-specific post-step hook.
            if hasattr(model, 'actions_after_step'):
                model.actions_after_step()
            
            loss_lst.append(loss.detach().cpu().data.numpy())

        return np.mean(loss_lst).item()

    def eval_termination(self, criterion: List[float]) -> bool:
        if len(criterion) > self.early_stop and utils.non_increasing(
            criterion[-self.early_stop :]
        ):
            return True
        elif len(criterion) - criterion.index(max(criterion)) > self.early_stop:
            return True
        return False

    def evaluate(
        self, dataset: BaseModel.Dataset, topks: list, metrics: list
    ) -> Dict[str, float]:
        """
        Evaluate the results for an input dataset.
        :return: result dict (key: metric@k)
        """
        top_idx, item_ids = self.predict_topk(dataset, topks)
        return self.evaluate_method_from_top_indices(top_idx, item_ids, topks, metrics)

    def predict_topk(self, dataset: BaseModel.Dataset, topks: list):
        """Streaming evaluation path that only keeps top-k indices on CPU."""
        dataset.model.eval()
        top_indices, item_ids = list(), list()
        max_topk = max(topks)

        use_persistent = (self.num_workers > 0) and (not self.pin_memory)
        dl = DataLoader(
            dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=dataset.collate_batch,
            pin_memory=self.pin_memory,
            persistent_workers=use_persistent,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )
        for batch in tqdm(dl, leave=False, ncols=100, mininterval=1, desc="Predict"):
            out_dict = None
            with torch.no_grad():
                if hasattr(dataset.model, "inference"):
                    prediction = dataset.model.inference(
                        utils.batch_to_gpu(batch, dataset.model.device)
                    )["prediction"]
                else:
                    out_dict = dataset.model(
                        utils.batch_to_gpu(batch, dataset.model.device), stage="infer"
                    )
                    prediction = out_dict["prediction"]
            prediction[:, -1] = -torch.inf
            topk_idx = torch.topk(prediction, k=max_topk, dim=1).indices
            top_indices.append(topk_idx.detach().cpu())
            item_ids.append(batch[ITEM_ID].detach().cpu())

            del prediction, topk_idx
            if out_dict is not None:
                del out_dict

        top_indices = torch.cat(top_indices, dim=0)
        item_ids = torch.cat(item_ids, dim=0)

        return top_indices, item_ids

    def predict(self, dataset: BaseModel.Dataset):
        """
        Generate predictions for a dataset.
        
        Args:
            dataset: Dataset to predict on
            
        Returns:
            predictions: Prediction scores
            item_ids: Target item IDs
        """
        dataset.model.eval()
        predictions, item_ids = list(), list()
        
        # Note: persistent_workers and pin_memory together cause memory/file handle leaks
        # in PyTorch. Only enable persistent_workers when pin_memory is disabled.
        use_persistent = (self.num_workers > 0) and (not self.pin_memory)
        dl = DataLoader(
            dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=dataset.collate_batch,
            pin_memory=self.pin_memory,
            persistent_workers=use_persistent,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )
        for batch in tqdm(dl, leave=False, ncols=100, mininterval=1, desc="Predict"):
            out_dict = None
            with torch.no_grad():
                if hasattr(dataset.model, "inference"):
                    prediction = dataset.model.inference(
                        utils.batch_to_gpu(batch, dataset.model.device)
                    )["prediction"]
                else:
                    out_dict = dataset.model(
                        utils.batch_to_gpu(batch, dataset.model.device), stage="infer"
                    )
                    prediction = out_dict["prediction"]
            predictions.append(prediction.detach().cpu())
            item_ids.append(batch[ITEM_ID].detach().cpu())

            if out_dict is not None:
                del out_dict
        
        predictions = torch.cat(predictions, dim=0)
        predictions[:, -1] = -torch.inf
        item_ids = torch.cat(item_ids, dim=0)
        return predictions, item_ids
    
    def print_res(self, dataset: BaseModel.Dataset) -> str:
        """
        Construct the final result string before/after training
        :return: test result string
        """
        result_dict = self.evaluate(dataset, self.topk, self.metrics)
        return "(" + utils.format_metric(result_dict) + ")"
