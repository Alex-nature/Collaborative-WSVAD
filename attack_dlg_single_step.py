import os
import json
import random

import numpy as np
import torch
import torch.nn.functional as F

# 为 DLG 的二阶梯度关闭高效 SDP attention，强制走 math 实现
if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

import utils.config as config
from utils.dataset import make_xd_dataloader, make_ucf_dataloader
from utils.model import Model
from utils.tools import (
    get_batch_label,
    get_prompt_text,
    build_negative_prompts,
    CLASM,
    NEG_LOSS_BCE,
    text_branch_regularization,
)


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def get_label_map(dataset: str) -> dict:
    if dataset == "ucf":
        return {
            "Normal": "normal",
            "Abuse": "abuse",
            "Arrest": "arrest",
            "Arson": "arson",
            "Assault": "assault",
            "Burglary": "burglary",
            "Explosion": "explosion",
            "Fighting": "fighting",
            "RoadAccidents": "roadAccidents",
            "Robbery": "robbery",
            "Shooting": "shooting",
            "Shoplifting": "shoplifting",
            "Stealing": "stealing",
            "Vandalism": "vandalism",
        }
    elif dataset == "xd":
        return {
            "A": "normal",
            "B1": "fighting",
            "B2": "shooting",
            "B4": "riot",
            "B5": "abuse",
            "B6": "car accident",
            "G": "explosion",
        }
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


def build_model(args, device: str) -> torch.nn.Module:
    model = Model(
        args.embed_dim,
        args.visual_length,
        args.prompt_prefix,
        args.prompt_postfix,
        args.visual_width,
        args.visual_head,
        args.visual_layers,
        device,
        use_tca=args.use_tca,
        tca_window_size=args.tca_window_size,
        tca_dropout=args.tca_dropout,
        use_distance_adj=args.use_distance_adj,
        tca_gamma=args.tca_gamma,
        tca_bias=args.tca_bias,
        tca_norm=args.tca_norm,
    ).to(device)

    model.initialize_separate_prompt_learners()

    if getattr(args, "checkpoint", None):
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint, strict=True)

    return model


def get_train_loaders(args):
    if args.dataset == "xd":
        train_loaders, _ = make_xd_dataloader(
            args.split_mode, args.clients_num, args.batch_size, args.visual_length
        )
    elif args.dataset == "ucf":
        train_loaders, _ = make_ucf_dataloader(
            args.split_mode, args.clients_num, args.batch_size, args.visual_length
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    return train_loaders


def prepare_one_real_step_batch(args, train_loaders, client_idx: int, device: str, label_map: dict):
    prompt_text = get_prompt_text(label_map)

    if args.dataset == "ucf":
        normal_loader, anomaly_loader = train_loaders[client_idx]

        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)

        normal_features, normal_label, normal_lengths = next(normal_iter)
        anomaly_features, anomaly_label, anomaly_lengths = next(anomaly_iter)

        real_x = torch.cat([normal_features, anomaly_features], dim=0).to(device).float()
        real_lengths = torch.cat([normal_lengths, anomaly_lengths], dim=0).to(device)

        text_labels_raw = list(normal_label) + list(anomaly_label)
        real_y = get_batch_label(
            text_labels_raw, prompt_text, label_map, args.dataset
        ).to(device).float()

        meta = {
            "raw_labels": text_labels_raw,
            "batch_size_effective": int(real_x.shape[0]),
            "client_idx": client_idx,
            "dataset": args.dataset,
            "note": "UCF single step = one normal batch + one anomaly batch concatenated",
        }

    elif args.dataset == "xd":
        train_loader = train_loaders[client_idx]
        train_iter = iter(train_loader)

        visual_feat, text_labels_raw, feat_lengths = next(train_iter)
        real_x = visual_feat.to(device).float()
        real_lengths = feat_lengths.to(device)
        real_y = get_batch_label(
            text_labels_raw, prompt_text, label_map, args.dataset
        ).to(device).float()

        meta = {
            "raw_labels": list(text_labels_raw),
            "batch_size_effective": int(real_x.shape[0]),
            "client_idx": client_idx,
            "dataset": args.dataset,
            "note": "XD single step = one dataloader batch",
        }

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    return real_x, real_y, real_lengths, meta


def get_all_trainable_parameters(model: torch.nn.Module):
    params = []
    names = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            params.append(p)
            names.append(name)
    return params, names


def get_used_parameters_from_loss(model: torch.nn.Module, loss: torch.Tensor):
    model.zero_grad(set_to_none=True)
    loss.backward(retain_graph=True)

    used_params = []
    used_names = []
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            used_params.append(p)
            used_names.append(name)

    model.zero_grad(set_to_none=True)
    return used_params, used_names


def forward_with_safe_sdp(
    model: torch.nn.Module,
    visual_features: torch.Tensor,
    prompt_text,
    feat_lengths: torch.Tensor,
    neg_prompt_text,
):
    if visual_features.is_cuda:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_math=True,
            enable_mem_efficient=False,
        ):
            outputs = model(
                visual_features,
                prompt_text,
                feat_lengths,
                is_training=True,
                neg_text=neg_prompt_text,
                return_text_features=True,
            )
    else:
        outputs = model(
            visual_features,
            prompt_text,
            feat_lengths,
            is_training=True,
            neg_text=neg_prompt_text,
            return_text_features=True,
        )
    return outputs


def compute_repo_training_loss(
    model: torch.nn.Module,
    visual_features: torch.Tensor,
    labels: torch.Tensor,
    feat_lengths: torch.Tensor,
    prompt_text,
    neg_prompt_text,
    device: str,
    dataset: str,
):
    logits_pos, logits_neg, text_features_pos, text_features_neg = forward_with_safe_sdp(
        model=model,
        visual_features=visual_features,
        prompt_text=prompt_text,
        feat_lengths=feat_lengths,
        neg_prompt_text=neg_prompt_text,
    )

    loss_ce = CLASM(logits_pos, labels, feat_lengths, device)
    loss_neg = NEG_LOSS_BCE(logits_pos, logits_neg, labels, feat_lengths, device)

    reg_lambda = 0.01 if dataset == "ucf" else 0.1
    loss_text_reg = text_branch_regularization(
        text_features_pos, text_features_neg, reg_lambda=reg_lambda, temperature=0.1
    )

    total_loss = loss_ce + loss_neg + loss_text_reg

    aux = {
        "loss_total": total_loss.detach().item(),
        "loss_ce": loss_ce.detach().item(),
        "loss_neg": loss_neg.detach().item(),
        "loss_text_reg": loss_text_reg.detach().item(),
        "logits_pos": logits_pos.detach(),
        "logits_neg": logits_neg.detach(),
    }
    return total_loss, aux


def compute_observed_gradients(
    model: torch.nn.Module,
    real_x: torch.Tensor,
    real_y: torch.Tensor,
    real_lengths: torch.Tensor,
    prompt_text,
    neg_prompt_text,
    device: str,
    dataset: str,
):
    model.zero_grad(set_to_none=True)

    loss, aux = compute_repo_training_loss(
        model=model,
        visual_features=real_x,
        labels=real_y,
        feat_lengths=real_lengths,
        prompt_text=prompt_text,
        neg_prompt_text=neg_prompt_text,
        device=device,
        dataset=dataset,
    )

    all_params, _ = get_all_trainable_parameters(model)
    used_params, used_param_names = get_used_parameters_from_loss(model, loss)

    print(f"All trainable parameter tensors: {len(all_params)}")
    print(f"Used parameter tensors in this real step: {len(used_params)}")

    observed_grads = torch.autograd.grad(
        loss,
        used_params,
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )
    observed_grads = [g.detach().clone() for g in observed_grads]

    return observed_grads, used_params, used_param_names, aux


def labels_from_parameter(
    label_param: torch.nn.Parameter,
    dataset: str,
    mode: str,
    known_label_tensor: torch.Tensor = None,
):
    if mode == "known":
        if known_label_tensor is None:
            raise ValueError("known mode requires known_label_tensor")
        return known_label_tensor

    if mode != "optimize":
        raise ValueError(f"Unsupported label mode: {mode}")

    if dataset == "ucf":
        return F.softmax(label_param, dim=1)

    if dataset == "xd":
        labels = torch.sigmoid(label_param)
        labels = labels + 1e-6
        return labels

    raise ValueError(f"Unsupported dataset: {dataset}")


def flatten_tensor_list(tensor_list):
    return torch.cat([t.reshape(-1) for t in tensor_list], dim=0)


def cosine_between_tensor_lists(list_a, list_b):
    flat_a = flatten_tensor_list(list_a)
    flat_b = flatten_tensor_list(list_b)
    return F.cosine_similarity(flat_a.unsqueeze(0), flat_b.unsqueeze(0), dim=1).item()


def filter_common_used_params_for_dummy(
    model: torch.nn.Module,
    used_params,
    used_param_names,
    dummy_x: torch.Tensor,
    dummy_y: torch.Tensor,
    real_lengths: torch.Tensor,
    prompt_text,
    neg_prompt_text,
    device: str,
    dataset: str,
):
    """
    防止 real step 用到的参数子集和 dummy step 用到的参数子集不完全一致。
    这里只保留两者公共且在 dummy graph 中也可求梯度的参数。
    """
    model.zero_grad(set_to_none=True)

    dummy_loss, _ = compute_repo_training_loss(
        model=model,
        visual_features=dummy_x,
        labels=dummy_y,
        feat_lengths=real_lengths,
        prompt_text=prompt_text,
        neg_prompt_text=neg_prompt_text,
        device=device,
        dataset=dataset,
    )

    dummy_grads = torch.autograd.grad(
        dummy_loss,
        used_params,
        create_graph=False,
        retain_graph=False,
        allow_unused=True,
    )

    common_params = []
    common_names = []
    common_indices = []

    for idx, (p, n, g) in enumerate(zip(used_params, used_param_names, dummy_grads)):
        if g is not None:
            common_params.append(p)
            common_names.append(n)
            common_indices.append(idx)

    return common_params, common_names, common_indices


def run_single_step_dlg(
    model: torch.nn.Module,
    observed_grads,
    used_params,
    used_param_names,
    real_x: torch.Tensor,
    real_y: torch.Tensor,
    real_lengths: torch.Tensor,
    prompt_text,
    neg_prompt_text,
    args,
    device: str,
):
    dummy_x = torch.nn.Parameter(torch.randn_like(real_x, device=device))

    if args.label_mode == "optimize":
        dummy_label_param = torch.nn.Parameter(torch.randn_like(real_y, device=device))
        optim_params = [dummy_x, dummy_label_param]
        init_dummy_y = labels_from_parameter(
            label_param=dummy_label_param,
            dataset=args.dataset,
            mode="optimize",
        )
    elif args.label_mode == "known":
        dummy_label_param = None
        optim_params = [dummy_x]
        init_dummy_y = real_y
    else:
        raise ValueError(f"Unsupported label_mode: {args.label_mode}")

    common_params, common_names, common_indices = filter_common_used_params_for_dummy(
        model=model,
        used_params=used_params,
        used_param_names=used_param_names,
        dummy_x=dummy_x,
        dummy_y=init_dummy_y,
        real_lengths=real_lengths,
        prompt_text=prompt_text,
        neg_prompt_text=neg_prompt_text,
        device=device,
        dataset=args.dataset,
    )

    if len(common_params) == 0:
        raise RuntimeError("No common parameter tensors between real step and dummy step.")

    observed_grads = [observed_grads[i] for i in common_indices]

    print(f"Common attacked parameter tensors after dummy-step filtering: {len(common_params)}")

    optimizer = torch.optim.LBFGS(
        optim_params,
        lr=args.dlg_lr,
        max_iter=args.dlg_max_iter,
        tolerance_grad=args.dlg_tolerance_grad,
        tolerance_change=args.dlg_tolerance_change,
        history_size=args.dlg_history_size,
        line_search_fn="strong_wolfe",
    )

    history = []

    def closure():
        optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)

        dummy_y = labels_from_parameter(
            label_param=dummy_label_param,
            dataset=args.dataset,
            mode=args.label_mode,
            known_label_tensor=real_y if args.label_mode == "known" else None,
        )

        dummy_loss, _ = compute_repo_training_loss(
            model=model,
            visual_features=dummy_x,
            labels=dummy_y,
            feat_lengths=real_lengths,
            prompt_text=prompt_text,
            neg_prompt_text=neg_prompt_text,
            device=device,
            dataset=args.dataset,
        )

        dummy_grads = torch.autograd.grad(
            dummy_loss,
            common_params,
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )

        grad_diff = 0.0
        for dg, og in zip(dummy_grads, observed_grads):
            grad_diff = grad_diff + ((dg - og) ** 2).sum()

        grad_diff.backward()
        return grad_diff

    for it in range(args.outer_iterations):
        loss_value = optimizer.step(closure)

        model.zero_grad(set_to_none=True)

        if args.label_mode == "optimize":
            current_dummy_y = labels_from_parameter(
                label_param=dummy_label_param,
                dataset=args.dataset,
                mode="optimize",
            )
        else:
            current_dummy_y = real_y

        current_dummy_loss, _ = compute_repo_training_loss(
            model=model,
            visual_features=dummy_x,
            labels=current_dummy_y,
            feat_lengths=real_lengths,
            prompt_text=prompt_text,
            neg_prompt_text=neg_prompt_text,
            device=device,
            dataset=args.dataset,
        )

        current_dummy_grads = torch.autograd.grad(
            current_dummy_loss,
            common_params,
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
        )

        with torch.no_grad():
            grad_mse = 0.0
            for dg, og in zip(current_dummy_grads, observed_grads):
                grad_mse += ((dg - og) ** 2).sum().item()

            feature_mse = F.mse_loss(dummy_x.detach(), real_x.detach()).item()
            feature_cos = F.cosine_similarity(
                dummy_x.detach().reshape(1, -1),
                real_x.detach().reshape(1, -1),
                dim=1,
            ).item()

            grad_cos = cosine_between_tensor_lists(
                [g.detach() for g in current_dummy_grads],
                observed_grads,
            )

            history.append(
                {
                    "iter": it + 1,
                    "grad_match_loss": float(loss_value.item() if torch.is_tensor(loss_value) else loss_value),
                    "grad_mse": float(grad_mse),
                    "grad_cosine": float(grad_cos),
                    "feature_mse": float(feature_mse),
                    "feature_cosine": float(feature_cos),
                }
            )

            if (it + 1) % args.print_every == 0 or it == 0 or (it + 1) == args.outer_iterations:
                print(
                    f"[DLG] iter={it+1:04d} "
                    f"grad_loss={history[-1]['grad_match_loss']:.6f} "
                    f"grad_cos={history[-1]['grad_cosine']:.6f} "
                    f"feat_mse={history[-1]['feature_mse']:.6f} "
                    f"feat_cos={history[-1]['feature_cosine']:.6f}"
                )

    with torch.no_grad():
        recovered_x = dummy_x.detach().clone()
        if args.label_mode == "optimize":
            recovered_y = labels_from_parameter(
                label_param=dummy_label_param,
                dataset=args.dataset,
                mode="optimize",
            ).detach().clone()
        else:
            recovered_y = real_y.detach().clone()

    result = {
        "recovered_x": recovered_x,
        "recovered_y": recovered_y,
        "history": history,
        "common_param_names": common_names,
    }
    return result


def evaluate_reconstruction(
    model: torch.nn.Module,
    real_x: torch.Tensor,
    rec_x: torch.Tensor,
    real_y: torch.Tensor,
    rec_y: torch.Tensor,
    real_lengths: torch.Tensor,
    prompt_text,
    neg_prompt_text,
    device: str,
    dataset: str,
):
    with torch.no_grad():
        real_loss, real_aux = compute_repo_training_loss(
            model=model,
            visual_features=real_x,
            labels=real_y,
            feat_lengths=real_lengths,
            prompt_text=prompt_text,
            neg_prompt_text=neg_prompt_text,
            device=device,
            dataset=dataset,
        )
        rec_loss, rec_aux = compute_repo_training_loss(
            model=model,
            visual_features=rec_x,
            labels=rec_y,
            feat_lengths=real_lengths,
            prompt_text=prompt_text,
            neg_prompt_text=neg_prompt_text,
            device=device,
            dataset=dataset,
        )

        feature_mse = F.mse_loss(rec_x, real_x).item()
        feature_cos = F.cosine_similarity(
            rec_x.reshape(1, -1),
            real_x.reshape(1, -1),
            dim=1,
        ).item()

        real_logits_pos = real_aux["logits_pos"].reshape(1, -1)
        rec_logits_pos = rec_aux["logits_pos"].reshape(1, -1)
        logits_cos = F.cosine_similarity(real_logits_pos, rec_logits_pos, dim=1).item()

        metrics = {
            "feature_mse": float(feature_mse),
            "feature_cosine": float(feature_cos),
            "real_loss_total": float(real_loss.item()),
            "rec_loss_total": float(rec_loss.item()),
            "positive_logits_cosine": float(logits_cos),
        }
    return metrics


def main():
    setup_seed(123456)

    parser = config.parser
    parser.add_argument("--client_idx", type=int, default=0, help="attacked client index")
    parser.add_argument(
        "--label_mode",
        type=str,
        default="optimize",
        choices=["optimize", "known"],
        help="optimize is standard DLG; known is easier ablation",
    )
    parser.add_argument("--outer_iterations", type=int, default=300)
    parser.add_argument("--dlg_lr", type=float, default=1.0)
    parser.add_argument("--dlg_max_iter", type=int, default=20)
    parser.add_argument("--dlg_history_size", type=int, default=100)
    parser.add_argument("--dlg_tolerance_grad", type=float, default=1e-10)
    parser.add_argument("--dlg_tolerance_change", type=float, default=1e-12)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="./dlg_single_step_results")
    parser.add_argument("--save_prefix", type=str, default="dlg_single_step")
    parser.add_argument("--export_npz", action="store_true")

    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")

    device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    ensure_dir(args.save_dir)

    label_map = get_label_map(args.dataset)
    prompt_text = get_prompt_text(label_map)
    neg_prompt_text, _ = build_negative_prompts(label_map)

    print("Building dataloaders...")
    train_loaders = get_train_loaders(args)

    if args.client_idx < 0 or args.client_idx >= len(train_loaders):
        raise IndexError(f"client_idx={args.client_idx} out of range, total clients={len(train_loaders)}")

    print("Building model...")
    model = build_model(args, device)
    model.eval()

    print("Preparing one real training step batch...")
    real_x, real_y, real_lengths, meta = prepare_one_real_step_batch(
        args=args,
        train_loaders=train_loaders,
        client_idx=args.client_idx,
        device=device,
        label_map=label_map,
    )

    print(f"Real step batch shape: x={tuple(real_x.shape)}, y={tuple(real_y.shape)}, lengths={tuple(real_lengths.shape)}")
    print(f"Raw labels: {meta['raw_labels']}")

    print("Computing observed gradients from the real step...")
    observed_grads, used_params, used_param_names, observed_aux = compute_observed_gradients(
        model=model,
        real_x=real_x,
        real_y=real_y,
        real_lengths=real_lengths,
        prompt_text=prompt_text,
        neg_prompt_text=neg_prompt_text,
        device=device,
        dataset=args.dataset,
    )

    print(f"Number of attacked parameter tensors before dummy filtering: {len(used_param_names)}")
    print("Starting single-step DLG inversion...")

    attack_result = run_single_step_dlg(
        model=model,
        observed_grads=observed_grads,
        used_params=used_params,
        used_param_names=used_param_names,
        real_x=real_x,
        real_y=real_y,
        real_lengths=real_lengths,
        prompt_text=prompt_text,
        neg_prompt_text=neg_prompt_text,
        args=args,
        device=device,
    )

    recovered_x = attack_result["recovered_x"]
    recovered_y = attack_result["recovered_y"]
    history = attack_result["history"]
    common_param_names = attack_result["common_param_names"]

    print("Evaluating reconstruction...")
    metrics = evaluate_reconstruction(
        model=model,
        real_x=real_x,
        rec_x=recovered_x,
        real_y=real_y,
        rec_y=recovered_y,
        real_lengths=real_lengths,
        prompt_text=prompt_text,
        neg_prompt_text=neg_prompt_text,
        device=device,
        dataset=args.dataset,
    )

    print("\n===== Final Reconstruction Metrics =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")

    result_prefix = (
        f"{args.save_prefix}_"
        f"{args.dataset}_"
        f"{args.split_mode}_"
        f"client{args.client_idx}_"
        f"bs{args.batch_size}_"
        f"label-{args.label_mode}"
    )

    pth_path = os.path.join(args.save_dir, result_prefix + ".pth")
    json_path = os.path.join(args.save_dir, result_prefix + ".json")

    save_obj = {
        "real_x": real_x.detach().cpu(),
        "real_y": real_y.detach().cpu(),
        "real_lengths": real_lengths.detach().cpu(),
        "recovered_x": recovered_x.detach().cpu(),
        "recovered_y": recovered_y.detach().cpu(),
        "history": history,
        "metrics": metrics,
        "meta": meta,
        "args": vars(args),
        "observed_loss_detail": {
            "loss_total": observed_aux["loss_total"],
            "loss_ce": observed_aux["loss_ce"],
            "loss_neg": observed_aux["loss_neg"],
            "loss_text_reg": observed_aux["loss_text_reg"],
        },
        "attacked_parameter_names_before_dummy_filter": used_param_names,
        "attacked_parameter_names_after_dummy_filter": common_param_names,
    }

    torch.save(save_obj, pth_path)

    json_obj = {
        "metrics": metrics,
        "meta": meta,
        "args": vars(args),
        "observed_loss_detail": {
            "loss_total": observed_aux["loss_total"],
            "loss_ce": observed_aux["loss_ce"],
            "loss_neg": observed_aux["loss_neg"],
            "loss_text_reg": observed_aux["loss_text_reg"],
        },
        "history": history,
        "attacked_parameter_names_before_dummy_filter": used_param_names,
        "attacked_parameter_names_after_dummy_filter": common_param_names,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)

    if args.export_npz:
        npz_path = os.path.join(args.save_dir, result_prefix + ".npz")
        np.savez_compressed(
            npz_path,
            real_x=real_x.detach().cpu().numpy(),
            real_y=real_y.detach().cpu().numpy(),
            real_lengths=real_lengths.detach().cpu().numpy(),
            recovered_x=recovered_x.detach().cpu().numpy(),
            recovered_y=recovered_y.detach().cpu().numpy(),
        )
        print(f"NPZ saved to: {npz_path}")

    print(f"PTH saved to: {pth_path}")
    print(f"JSON saved to: {json_path}")


if __name__ == "__main__":
    main()