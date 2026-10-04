# -*- coding: UTF-8 -*-

from datetime import datetime
import argparse
import hashlib
import logging
import os
import pickle
import sys

# Set CUDA_VISIBLE_DEVICES before importing torch.
_gpu_parser = argparse.ArgumentParser(add_help=False)
_gpu_parser.add_argument("--gpu", type=str, default="0")
_gpu_args, _ = _gpu_parser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_args.gpu

import torch

sys.path.append("..")

from helpers import *
from models import *
from utils import utils
from utils.corpus_cache import build_corpus_cache_path
from utils.data_roots import validate_dataset_path


def parse_global_args(parser):
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="Set CUDA_VISIBLE_DEVICES. Use an empty string for CPU.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=logging.INFO,
        help="Logging level: 0, 10, ..., 50.",
    )
    parser.add_argument("--log_file", type=str, default="", help="Logging file path.")
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Optional random seed for NumPy and PyTorch.",
    )
    parser.add_argument(
        "--load",
        type=int,
        default=0,
        help="Whether to load a model checkpoint before training/evaluation.",
    )
    parser.add_argument(
        "--load_model_path",
        type=str,
        default=None,
        help="Checkpoint path for --load.",
    )
    parser.add_argument(
        "--train",
        type=int,
        default=1,
        help="Whether to train the model.",
    )
    parser.add_argument(
        "--regenerate",
        type=int,
        default=0,
        help="Whether to regenerate cached corpus objects.",
    )
    parser.add_argument(
        "--data_profile",
        type=str,
        default="",
        help="Named dataset profile, e.g. amazon_abs_v3_rearec.",
    )
    return parser


def _resolve_device(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.device = torch.device("cpu")
    if args.gpu != "" and torch.cuda.is_available():
        args.device = torch.device("cuda")
    args.is_main_process = True


def _configure_root_logger(log_file, level, stream=None):
    basic_config_kwargs = {
        "level": level,
        "force": True,
    }
    if log_file:
        basic_config_kwargs["filename"] = log_file
    logging.basicConfig(**basic_config_kwargs)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    if stream is None:
        stream = sys.stdout
    root_logger.addHandler(logging.StreamHandler(stream))
    return root_logger


def main():
    logging.info("-" * 45 + " BEGIN: " + utils.get_time() + " " + "-" * 45)
    exclude = [
        "check_epoch",
        "log_file",
        "model_path",
        "path",
        "pin_memory",
        "load",
        "regenerate",
        "sep",
        "train",
        "verbose",
        "metric",
        "test_epoch",
        "buffer",
        "random_seed",
    ]
    logging.info(utils.format_arg_str(args, exclude_lst=exclude))

    if args.random_seed is not None:
        utils.init_seed(args.random_seed)
    _resolve_device(args)
    logging.info("Device: {}".format(args.device))

    try:
        corpus_path = build_corpus_cache_path(
            args.path,
            args.dataset,
            reader_name.__name__,
        )
        if not args.regenerate and os.path.exists(corpus_path):
            logging.info("Load corpus from {}".format(corpus_path))
            with open(corpus_path, "rb") as corpus_file:
                corpus = pickle.load(corpus_file)
        else:
            corpus = reader_name(args)
            logging.info("Save corpus to {}".format(corpus_path))
            utils.check_dir(corpus_path)
            with open(corpus_path, "wb") as corpus_file:
                pickle.dump(corpus, corpus_file)

        model = model_name(args, corpus).to(args.device)
        logging.info("#params: {}".format(model.count_variables()))
        logging.info(model)

        data_dict = {}
        for phase in ["train", "valid", "test"]:
            data_dict[phase] = model_name.Dataset(model, corpus, phase)
            data_dict[phase].prepare()

        runner = runner_name(args)
        if args.load > 0:
            model.load_model(args.load_model_path)

        logging.info("Valid Before Training: " + runner.print_res(data_dict["valid"]))
        logging.info("Test Before Training: " + runner.print_res(data_dict["test"]))

        if args.train > 0:
            runner.train(data_dict)

        eval_res = runner.print_res(data_dict["test"])
        logging.info(os.linesep + "Test After Training: " + eval_res)
        model.actions_after_train()
        logging.info(os.linesep + "-" * 45 + " END: " + utils.get_time() + " " + "-" * 45)
    except Exception:
        logging.exception("Run failed.")
        raise


if __name__ == "__main__":
    init_parser = argparse.ArgumentParser(description="Model")
    init_parser.add_argument(
        "--model_name",
        type=str,
        default="SASRec",
        help="Choose a model to run.",
    )
    init_args, _ = init_parser.parse_known_args()

    model_name = eval("{0}.{0}".format(init_args.model_name))
    reader_name = eval("{0}.{0}".format(model_name.reader))
    runner_name = eval("{0}.{0}".format(model_name.runner))

    parser = argparse.ArgumentParser(description="")
    parser = parse_global_args(parser)
    parser = reader_name.parse_data_args(parser)
    parser = runner_name.parse_runner_args(parser)
    parser = model_name.parse_model_args(parser)
    args, _ = parser.parse_known_args()
    try:
        inferred_profile = validate_dataset_path(
            args.path,
            args.dataset,
            getattr(args, "data_profile", ""),
        )
    except ValueError as exc:
        parser.error(str(exc))
    if not getattr(args, "data_profile", "") and inferred_profile:
        args.data_profile = inferred_profile

    log_args = [init_args.model_name, args.dataset]
    for arg in ["lr", "l2"] + model_name.extra_log_args:
        log_args.append(arg + "=" + str(getattr(args, arg)))

    config_str = "__".join(log_args)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    log_file_name = f"{init_args.model_name}__{args.dataset}__{config_hash}__{timestamp}"
    if args.log_file == "":
        args.log_file = "log/{}/{}.log".format(init_args.model_name, log_file_name)
    if args.model_path == "":
        args.model_path = "save_model/{}/{}.pt".format(
            init_args.model_name,
            log_file_name,
        )
    args.model_name = init_args.model_name

    utils.check_dir(args.log_file)
    _configure_root_logger(args.log_file, args.verbose, stream=sys.stdout)
    logging.info(init_args)

    main()
