import argparse
import datetime
import json
import os
import pickle
import sys
import time
from collections import OrderedDict
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models import create_model
from timm.utils import accuracy

import modeling_finetune  # noqa: F401; registers VideoMAE models with timm.
import utils
from datasets import build_dataset
from engine_for_finetuning import merge, merge_plus, train_one_epoch, validation_one_epoch
from mixup import Mixup
from optim_factory import LayerDecayValueAssigner, create_optimizer
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import multiple_samples_collate


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


class StdoutStderrTee:
    def __init__(self, log_path):
        self.log_path = log_path
        self._fp = None
        self._stdout = None
        self._stderr = None

    def __enter__(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self._fp = open(self.log_path, "a", encoding="utf-8")
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = TeeStream(self._stdout, self._fp)
        sys.stderr = TeeStream(self._stderr, self._fp)
        print(f"[Logging] Console output is also written to: {self.log_path}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self._fp is not None:
                self._fp.flush()
        finally:
            sys.stdout = self._stdout
            sys.stderr = self._stderr
            if self._fp is not None:
                self._fp.close()


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location)
    except pickle.UnpicklingError as exc:
        msg = str(exc)
        if "Weights only load failed" in msg or "weights_only" in msg:
            print(f"[safe_torch_load] Retrying trusted checkpoint with weights_only=False: {path}")
            return torch.load(path, map_location=map_location, weights_only=False)
        raise


def get_args():
    parser = argparse.ArgumentParser(
        "cross-CholeAct VideoMAE fine-tuning and evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_path", required=True, type=str, help="folder containing train.csv, val.csv, test.csv")
    parser.add_argument("--finetune", required=True, type=str, help="VideoMAE checkpoint used to initialize the model")
    parser.add_argument("--output_dir", default="./output_finetune_8cls", type=str, help="checkpoint/result output folder")
    parser.add_argument("--nb_classes", default=8, type=int, help="number of action classes")
    parser.add_argument("--batch_size", default=4, type=int, help="mini-batch size")
    parser.add_argument("--num_workers", default=32, type=int, help="DataLoader worker count")
    parser.add_argument("--epochs", default=30, type=int, help="fine-tuning epochs")
    parser.add_argument("--device", default="cuda:0", type=str, help="device used for training/evaluation")
    parser.add_argument("--save_ckpt_freq", default=10, type=int, help="checkpoint save frequency")
    parser.add_argument("--log_dir", default="./logs_finetune", type=str, help="TensorBoard log folder")
    parser.add_argument("--eval", action="store_true", help="run evaluation only")
    parser.add_argument(
        "--extract_eval_features",
        action="store_true",
        help="additionally dump penultimate VideoMAE features for domain_gap_analysis.py",
    )
    parser.add_argument("--feature_dump_level", default="video", choices=["video", "view"], help="feature dump granularity")
    parser.add_argument("--feature_dump_dir", default="", type=str, help="feature dump folder")
    parser.add_argument("--feature_dataset_source", default="private", type=str, help="dataset tag stored in feature npz")
    parser.add_argument("--feature_split_name", default="test", type=str, help="split tag stored in feature npz")
    parser.add_argument("--feature_save_logits", action="store_true", help="also save logits in the feature npz")
    args = parser.parse_args()
    apply_fixed_defaults(args)
    return args


def apply_fixed_defaults(args):
    defaults = {
        "model": "vit_base_patch16_224",
        "data_set": "NTUH",
        "tubelet_size": 2,
        "input_size": 224,
        "fc_drop_rate": 0.0,
        "drop": 0.0,
        "attn_drop_rate": 0.0,
        "drop_path": 0.1,
        "model_key": "model|module",
        "model_prefix": "",
        "init_scale": 0.001,
        "use_checkpoint": False,
        "use_mean_pooling": True,
        "num_segments": 1,
        "num_frames": 16,
        "sampling_rate": 4,
        "short_side_size": 224,
        "test_num_segment": 5,
        "test_num_crop": 3,
        "crop_pct": None,
        "opt": "adamw",
        "opt_eps": 1e-8,
        "opt_betas": None,
        "clip_grad": None,
        "momentum": 0.9,
        "weight_decay": 0.05,
        "weight_decay_end": None,
        "lr": 1e-3,
        "layer_decay": 0.75,
        "warmup_lr": 1e-6,
        "min_lr": 1e-6,
        "warmup_epochs": 5,
        "warmup_steps": -1,
        "color_jitter": 0.4,
        "num_sample": 2,
        "aa": "rand-m7-n4-mstd0.5-inc1",
        "smoothing": 0.1,
        "train_interpolation": "bicubic",
        "reprob": 0.25,
        "remode": "pixel",
        "recount": 1,
        "resplit": False,
        "mixup": 0.8,
        "cutmix": 1.0,
        "cutmix_minmax": None,
        "mixup_prob": 1.0,
        "mixup_switch_prob": 0.5,
        "mixup_mode": "batch",
        "imagenet_default_mean_and_std": True,
        "disable_eval_during_finetuning": False,
        "save_ckpt": True,
        "auto_resume": False,
        "resume": "",
        "start_epoch": 0,
        "seed": 0,
        "pin_mem": True,
        "dist_eval": False,
        "world_size": 1,
        "local_rank": -1,
        "dist_on_itp": False,
        "dist_url": "env://",
        "enable_deepspeed": False,
        "update_freq": 1,
        "feature_max_samples": 0,
    }
    if args.eval:
        defaults["log_dir"] = None
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    if not args.feature_dump_dir and args.output_dir:
        args.feature_dump_dir = os.path.join(args.output_dir, "feature_dumps")
    args.console_log = os.path.join(args.output_dir, "eval_console.log" if args.eval else "train_console.log")


def load_finetune_checkpoint(model, args):
    if args.finetune.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(args.finetune, map_location="cpu", check_hash=True)
    else:
        checkpoint = safe_torch_load(args.finetune, map_location="cpu")

    print(f"Load ckpt from {args.finetune}")
    checkpoint_model = None
    if isinstance(checkpoint, dict):
        for model_key in args.model_key.split("|"):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print(f"Load state_dict by model_key = {model_key}")
                break
    if checkpoint_model is None:
        checkpoint_model = checkpoint

    state_dict = model.state_dict()
    for key in ["head.weight", "head.bias"]:
        if key in checkpoint_model and checkpoint_model[key].shape != state_dict[key].shape:
            print(f"Removing key {key} from pretrained checkpoint")
            del checkpoint_model[key]

    new_dict = OrderedDict()
    for key, value in checkpoint_model.items():
        if key.startswith("backbone."):
            new_dict[key[9:]] = value
        elif key.startswith("encoder."):
            new_dict[key[8:]] = value
        else:
            new_dict[key] = value
    checkpoint_model = new_dict

    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens) // (args.num_frames // model.patch_embed.tubelet_size)) ** 0.5)
        new_size = int((num_patches // (args.num_frames // model.patch_embed.tubelet_size)) ** 0.5)
        if orig_size != new_size:
            print(f"Position interpolate from {orig_size}x{orig_size} to {new_size}x{new_size}")
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, args.num_frames // model.patch_embed.tubelet_size, orig_size, orig_size, embedding_size)
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(
                -1, args.num_frames // model.patch_embed.tubelet_size, new_size, new_size, embedding_size
            )
            checkpoint_model["pos_embed"] = torch.cat((extra_tokens, pos_tokens.flatten(1, 3)), dim=1)

    utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)


def _ensure_list_str(x, n):
    if x is None:
        return [""] * n
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        return [str(v) for v in x]
    return [str(x)] * n


def _as_int_str(x):
    if isinstance(x, torch.Tensor):
        return str(int(x.detach().cpu().item()))
    try:
        return str(int(x))
    except Exception:
        return str(x)


def _parse_eval_batch(batch):
    if len(batch) >= 5:
        return batch[:5]
    raise ValueError(f"Expected test batch with videos, target, id, chunk_nb, split_nb; got {type(batch)}")


def _extract_logits_and_features(model, videos):
    target_model = model.module if hasattr(model, "module") else model
    if hasattr(target_model, "forward_features"):
        feat = target_model.forward_features(videos)
        if isinstance(feat, (list, tuple)):
            feat = feat[0]
        if isinstance(feat, torch.Tensor) and feat.ndim == 3:
            feat = feat.mean(dim=1)
        logits = target_model.head(target_model.fc_dropout(feat)) if hasattr(target_model, "fc_dropout") else target_model.head(feat)
        return logits, feat

    holder = {}
    handle = None
    if hasattr(target_model, "head"):
        def _hook(_module, inputs, _output):
            if isinstance(inputs, (list, tuple)) and len(inputs) > 0:
                holder["feat"] = inputs[0].detach()
        handle = target_model.head.register_forward_hook(_hook)
    logits = model(videos)
    if handle is not None:
        handle.remove()
    return logits, holder.get("feat", logits.detach())


def _save_eval_feature_npz(args, features, logits, labels, preds, video_ids, clip_centers, clip_uids, level):
    os.makedirs(args.feature_dump_dir, exist_ok=True)
    rank = utils.get_rank()
    rank_suffix = f"_rank{rank}" if utils.get_world_size() > 1 else ""
    out_path = os.path.join(
        args.feature_dump_dir, f"{args.feature_split_name}_{args.feature_dataset_source}_{level}{rank_suffix}_features.npz")
    save_kwargs = {
        "features": features.astype(np.float32),
        "labels": labels.astype(np.int64),
        "predictions": preds.astype(np.int64),
        "preds": preds.astype(np.int64),
        "video_ids": np.asarray(video_ids, dtype=object),
        "video_id": np.asarray(video_ids, dtype=object),
        "clip_center": np.asarray(clip_centers, dtype=object),
        "clip_uid": np.asarray(clip_uids, dtype=object),
        "dataset_source": np.asarray([str(args.feature_dataset_source)] * len(labels), dtype=object),
        "split_name": np.asarray([str(args.feature_split_name)] * len(labels), dtype=object),
        "feature_dump_level": np.asarray([level], dtype=object),
        "num_classes": np.asarray([int(args.nb_classes)], dtype=np.int64),
    }
    if args.feature_save_logits:
        save_kwargs["logits"] = logits.astype(np.float32)
    np.savez_compressed(out_path, **save_kwargs)
    print(f"[FeatureDump] saved: {out_path} | level={level} | n={len(labels)} | feat_dim={features.shape[-1]}")


def final_test_with_optional_feature_dump(data_loader, model, device, file, args):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    model.eval()
    final_result = []

    save_features = bool(args.extract_eval_features)
    feature_level = str(args.feature_dump_level).lower()
    per_video = {}
    view_features, view_logits, view_labels, view_preds = [], [], [], []
    view_video_ids, view_clip_centers, view_clip_uids = [], [], []

    with torch.no_grad():
        for batch in metric_logger.log_every(data_loader, 10, "Test:"):
            videos, target, ids, chunk_nb, split_nb = _parse_eval_batch(batch)
            videos = videos.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                if save_features:
                    output, feat = _extract_logits_and_features(model, videos)
                else:
                    output = model(videos)
                    feat = None
                loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            batch_size = videos.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

            ids_list = _ensure_list_str(ids, batch_size)
            chunk_list = _ensure_list_str(chunk_nb, batch_size)
            split_list = _ensure_list_str(split_nb, batch_size)
            target_np = target.detach().cpu().numpy().astype(np.int64)
            output_np = output.detach().float().cpu().numpy()
            feat_np = feat.detach().float().cpu().numpy() if save_features else None

            for i in range(output.size(0)):
                vid = ids_list[i]
                chunk_i = _as_int_str(chunk_list[i])
                split_i = _as_int_str(split_list[i])
                label_i = int(target_np[i])
                final_result.append(f"{vid} {str(output_np[i].tolist())} {label_i} {chunk_i} {split_i}\n")

                if save_features:
                    pos_key = f"{chunk_i}_{split_i}"
                    clip_center = f"chunk{chunk_i}_crop{split_i}"
                    clip_uid = f"{vid}__chunk{chunk_i}__crop{split_i}"
                    pred_i = int(np.argmax(output_np[i]))
                    if feature_level == "view":
                        view_features.append(feat_np[i])
                        view_logits.append(output_np[i])
                        view_labels.append(label_i)
                        view_preds.append(pred_i)
                        view_video_ids.append(vid)
                        view_clip_centers.append(clip_center)
                        view_clip_uids.append(clip_uid)
                    else:
                        if vid not in per_video:
                            per_video[vid] = {"feat": [], "logits": [], "label": label_i, "pos": set()}
                        if pos_key in per_video[vid]["pos"]:
                            continue
                        per_video[vid]["pos"].add(pos_key)
                        per_video[vid]["feat"].append(feat_np[i])
                        per_video[vid]["logits"].append(output_np[i])
                        per_video[vid]["label"] = label_i

    with open(file, "w", encoding="utf-8") as fp:
        fp.write(f"{acc1}, {acc5}\n")
        fp.writelines(final_result)

    metric_logger.synchronize_between_processes()
    print("* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}".format(
        top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss
    ))

    if save_features:
        if feature_level == "view" and view_labels:
            _save_eval_feature_npz(
                args,
                np.stack(view_features, axis=0),
                np.stack(view_logits, axis=0),
                np.asarray(view_labels),
                np.asarray(view_preds),
                view_video_ids,
                view_clip_centers,
                view_clip_uids,
                "view",
            )
        elif feature_level == "video" and per_video:
            video_ids, features, logits, labels, preds, clip_centers, clip_uids = [], [], [], [], [], [], []
            for vid, rec in per_video.items():
                logit_avg = np.mean(np.stack(rec["logits"], axis=0), axis=0)
                video_ids.append(vid)
                features.append(np.mean(np.stack(rec["feat"], axis=0), axis=0))
                logits.append(logit_avg)
                labels.append(int(rec["label"]))
                preds.append(int(np.argmax(logit_avg)))
                clip_centers.append("merged_views")
                clip_uids.append(f"{vid}__merged_views")
            _save_eval_feature_npz(
                args,
                np.stack(features, axis=0),
                np.stack(logits, axis=0),
                np.asarray(labels),
                np.asarray(preds),
                video_ids,
                clip_centers,
                clip_uids,
                "video",
            )
        else:
            print("[FeatureDump] no samples found; skip dump.")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def write_result_txt(output_dir, video_ids, predictions, truelabels):
    result_path = os.path.join(output_dir, "result.txt")
    with open(result_path, "w", encoding="utf-8") as fp:
        for vid, pred, lab in zip(video_ids, predictions, truelabels):
            fp.write(f"{vid} {pred} {lab}\n")
    print(f"[Result] wrote result.txt: {result_path}")


def make_dataloaders(args):
    dataset_train, args.nb_classes = build_dataset(is_train=True, test_mode=False, args=args)
    dataset_val, _ = build_dataset(is_train=False, test_mode=False, args=args)
    dataset_test, _ = build_dataset(is_train=False, test_mode=True, args=args)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    collate_func = partial(multiple_samples_collate, fold=False) if args.num_sample > 1 else None
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=collate_func,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    return dataset_train, dataset_val, dataset_test, data_loader_train, data_loader_val, data_loader_test


def create_videomae_model(args):
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        all_frames=args.num_frames * args.num_segments,
        tubelet_size=args.tubelet_size,
        fc_drop_rate=args.fc_drop_rate,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        drop_block_rate=None,
        use_checkpoint=args.use_checkpoint,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
    )
    patch_size = model.patch_embed.patch_size
    args.window_size = (args.num_frames // 2, args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size
    load_finetune_checkpoint(model, args)
    return model


def evaluate(args, model, data_loader_test, dataset_test, device):
    global_rank = utils.get_rank()
    num_tasks = utils.get_world_size()
    preds_file = os.path.join(args.output_dir, f"{global_rank}.txt")
    final_test_with_optional_feature_dump(data_loader_test, model, device, preds_file, args)
    if args.distributed:
        torch.distributed.barrier()
    if global_rank == 0:
        print("Start merging results...")
        final_top1, final_top5, f1_per_class, f1_macro = merge(args.output_dir, num_tasks)
        _, _, video_ids, predictions, truelabels = merge_plus(args.output_dir, num_tasks)
        write_result_txt(args.output_dir, video_ids, predictions, truelabels)
        print(
            f"Accuracy of the network on the {len(dataset_test)} test videos: "
            f"Top-1: {final_top1:.2f}%, Top-5: {final_top5:.2f}%, Macro-F1: {f1_macro:.4f}"
        )
        print(f"Test F1 per class: {np.array2string(f1_per_class, precision=4, suppress_small=True)}")
        log_stats = {
            "Final top-1": final_top1,
            "Final Top-5": final_top5,
            "Final F1-macro": float(f1_macro),
            "Final F1-per-class": np.nan_to_num(f1_per_class, nan=-1.0).tolist(),
        }
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as fp:
            fp.write(json.dumps(log_stats) + "\n")


def main(args):
    utils.init_distributed_mode(args)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)

    tee_ctx = StdoutStderrTee(args.console_log) if utils.is_main_process() else nullcontext()
    tee_ctx.__enter__()
    try:
        print(args)
        device = torch.device(args.device)
        seed = args.seed + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        cudnn.benchmark = True

        dataset_train, dataset_val, dataset_test, data_loader_train, data_loader_val, data_loader_test = make_dataloaders(args)

        log_writer = utils.TensorboardLogger(log_dir=args.log_dir) if utils.is_main_process() and args.log_dir and not args.eval else None
        mixup_fn = None
        if args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None:
            print("Mixup is activated!")
            mixup_fn = Mixup(
                mixup_alpha=args.mixup,
                cutmix_alpha=args.cutmix,
                cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob,
                switch_prob=args.mixup_switch_prob,
                mode=args.mixup_mode,
                label_smoothing=args.smoothing,
                num_classes=args.nb_classes,
            )

        model = create_videomae_model(args)
        model.to(device)
        model_without_ddp = model
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Model = %s" % str(model_without_ddp))
        print("number of params:", n_parameters)

        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
            model_without_ddp = model.module

        if args.eval:
            evaluate(args, model, data_loader_test, dataset_test, device)
            return

        total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
        num_training_steps_per_epoch = len(dataset_train) // total_batch_size
        args.lr = args.lr * total_batch_size / 256
        args.min_lr = args.min_lr * total_batch_size / 256
        args.warmup_lr = args.warmup_lr * total_batch_size / 256
        print("LR = %.8f" % args.lr)
        print("Batch size = %d" % total_batch_size)
        print("Number of training examples = %d" % len(dataset_train))
        print("Number of training steps per epoch = %d" % num_training_steps_per_epoch)

        num_layers = model_without_ddp.get_num_layers()
        assigner = LayerDecayValueAssigner([args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)])
        skip_weight_decay_list = model_without_ddp.no_weight_decay()
        optimizer = create_optimizer(
            args,
            model_without_ddp,
            skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id,
            get_layer_scale=assigner.get_scale,
        )
        loss_scaler = NativeScaler()

        lr_schedule_values = utils.cosine_scheduler(
            args.lr,
            args.min_lr,
            args.epochs,
            num_training_steps_per_epoch,
            warmup_epochs=args.warmup_epochs,
            warmup_steps=args.warmup_steps,
        )
        if args.weight_decay_end is None:
            args.weight_decay_end = args.weight_decay
        wd_schedule_values = utils.cosine_scheduler(args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)

        if mixup_fn is not None:
            criterion = SoftTargetCrossEntropy()
        elif args.smoothing > 0.0:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss()
        print("criterion = %s" % str(criterion))

        print(f"Start training for {args.epochs} epochs")
        start_time = time.time()
        best_f1_macro = float("-inf")
        best_acc1 = float("-inf")
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)
            if log_writer is not None:
                log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
            train_stats = train_one_epoch(
                model,
                criterion,
                data_loader_train,
                optimizer,
                device,
                epoch,
                loss_scaler,
                args.clip_grad,
                None,
                mixup_fn,
                log_writer=log_writer,
                start_steps=epoch * num_training_steps_per_epoch,
                lr_schedule_values=lr_schedule_values,
                wd_schedule_values=wd_schedule_values,
                num_training_steps_per_epoch=num_training_steps_per_epoch,
                update_freq=args.update_freq,
            )
            if args.output_dir and ((epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs):
                utils.save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, None)

            val_stats, _ = validation_one_epoch(data_loader_val, model, device)
            val_f1_per_class = val_stats["f1_per_class"]
            val_f1_macro = val_stats["f1_macro"]
            print(
                f"Accuracy of the network on the {len(dataset_val)} val videos: "
                f"{val_stats['acc1']:.1f}% | Macro-F1: {val_f1_macro:.4f}"
            )
            print(f"Val F1 per class: {np.array2string(val_f1_per_class, precision=4, suppress_small=True)}")

            if val_f1_macro > best_f1_macro or (np.isclose(val_f1_macro, best_f1_macro) and val_stats["acc1"] > best_acc1):
                best_f1_macro = float(val_f1_macro)
                best_acc1 = float(val_stats["acc1"])
                utils.save_model(args, "best", model, model_without_ddp, optimizer, loss_scaler, None)

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "val_loss": float(val_stats["loss"]),
                "val_acc1": float(val_stats["acc1"]),
                "val_acc5": float(val_stats["acc5"]),
                "val_f1_macro": float(val_f1_macro),
                "val_f1_per_class": np.nan_to_num(val_f1_per_class, nan=-1.0).tolist(),
                "best_val_f1_macro": float(best_f1_macro),
                "best_val_acc1_tiebreak": float(best_acc1),
                "epoch": epoch,
                "n_parameters": n_parameters,
            }
            if utils.is_main_process():
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as fp:
                    fp.write(json.dumps(log_stats) + "\n")

        best_ckpt_path = os.path.join(args.output_dir, "checkpoint-best.pth")
        if os.path.exists(best_ckpt_path):
            print(f"Loading best checkpoint for final test: {best_ckpt_path}")
            best_checkpoint = safe_torch_load(best_ckpt_path, map_location="cpu")
            best_checkpoint_model = best_checkpoint["model"] if isinstance(best_checkpoint, dict) and "model" in best_checkpoint else best_checkpoint
            utils.load_state_dict(model_without_ddp, best_checkpoint_model, prefix=args.model_prefix)
            model.to(device)
        else:
            print("checkpoint-best.pth not found; using the last in-memory model for final test.")
        evaluate(args, model, data_loader_test, dataset_test, device)

        total_time = time.time() - start_time
        print("Training time {}".format(str(datetime.timedelta(seconds=int(total_time)))))
    finally:
        tee_ctx.__exit__(None, None, None)


if __name__ == "__main__":
    opts = get_args()
    Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
