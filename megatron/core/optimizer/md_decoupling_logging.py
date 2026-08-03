# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Distributed TensorBoard/W&B statistics for MDDecoupling gains.

The computation has three stages:

1. :func:`collect_md_gain_stats` accumulates element-level gain distributions and saturation.
2. :func:`_accumulate_md_matrix_stats` reconstructs TP-sharded matrix gains, then
   :func:`_accumulate_reduced_matrix_batch` computes per-matrix RMS, scale, gauge, and
   effective-weight sparsity values.
3. :func:`_append_md_gain_stats` converts the globally reduced sums into logged averages.
"""

import math
from typing import Dict, List, TYPE_CHECKING

import torch

from megatron.core.utils import get_pg_rank, get_pg_size

from .layer_wise_optimizer import LayerWiseDistributedOptimizer

if TYPE_CHECKING:
    from .md_decoupling import MDDecoupling


_GAIN_AXES = ("row", "col", "flat")
_MATRIX_SQUARE_SUM, _MATRIX_ELEMENT_COUNT, _MATRIX_LOG_SUM, _WEIGHT_SQUARE_SUM, _WEIGHT_ELEMENT_COUNT, _MATRIX_NEAR_ZERO_START = range(6)  # fmt: skip

# Columns of ``totals``: element-weighted effective-gain distribution statistics.
_GAIN_SUM, _GAIN_SQUARE_SUM, _GAIN_ELEMENT_COUNT, _GAIN_SOFTPLUS_COUNT, _GAIN_SATURATED_COUNT, _GAIN_STAT_COUNT = range(6)  # fmt: skip

# Columns of ``matrix_axis_totals``: equal-weighted per-matrix effective RMS.
_MATRIX_RMS_SUM, _MATRIX_COUNT, _MATRIX_AXIS_STAT_COUNT = range(3)

# Fixed columns of ``matrix_totals``; one sparsity-sum column per threshold follows these.
_SCALE_SUM, _PAIR_COUNT, _COMBINED_LOG_SCALE_SUM, _ROW_COL_IMBALANCE_SUM, _SOFTPLUS_PAIR_COUNT, _SPARSITY_MATRIX_COUNT, _PARAM_RMS_SUM, _PARAM_RMS_COUNT, _SPARSITY_SUM_START = range(9)  # fmt: skip

_DEFAULT_SPARSITY_THRESHOLDS = (1e-20, 1e-10, 1e-30)
_GAIN_FAMILIES = (
    "router",
    "embedding",
    "output",
    "attention-in",
    "attention-out",
    "expert-in",
    "expert-out",
    "moe-latent-in",
    "moe-latent-out",
    "dense-mlp-in",
    "dense-mlp-out",
    "unclassified",
)


def _gain_log_family(name: str, param: torch.Tensor) -> str:
    """Assign a stable, low-cardinality logging family while the parameter name is available."""
    if getattr(param, "is_router", False):
        return "router"
    if getattr(param, "is_md_embedding_parameter", False):
        return "embedding"
    if getattr(param, "is_md_output_parameter", False):
        return "output"

    is_out = getattr(param, "is_out_proj", False)
    if "fc1_latent_proj" in name:
        return "moe-latent-in"
    if "fc2_latent_proj" in name:
        return "moe-latent-out"
    if "experts" in name:
        return "expert-out" if is_out else "expert-in"
    if "attention" in name:
        return "attention-out" if is_out else "attention-in"
    if ".mlp." in name:
        return "dense-mlp-out" if is_out else "dense-mlp-in"
    return "unclassified"


def _include_gain_in_global_stats(
    md_optimizer: "MDDecoupling",
    param: torch.Tensor,
    axis: str,
    dp_state_is_sharded: bool,
) -> bool:
    """Return whether this rank owns a unique logical copy of a gain tensor."""
    pg_collection = md_optimizer.pg_collection
    if pg_collection is None:
        return True

    is_expert = getattr(param, "expert_tp", False)
    if not dp_state_is_sharded:
        dp_group = (
            getattr(pg_collection, "expt_dp", None)
            if is_expert
            else getattr(pg_collection, "dp_cp", None)
        )
        if dp_group is not None and get_pg_rank(dp_group) != 0:
            return False

    retained_dim = {"row": 0, "col": 1, "flat": None}[axis]
    gain_is_tp_sharded = (
        retained_dim is not None
        and getattr(param, "partition_dim", None) == retained_dim
    )
    if not gain_is_tp_sharded:
        tp_group = (
            getattr(pg_collection, "expt_tp", None)
            if is_expert
            else getattr(pg_collection, "tp", None)
        )
        if tp_group is not None and get_pg_rank(tp_group) != 0:
            return False
    return True


def _include_param_in_matrix_stats(
    md_optimizer: "MDDecoupling", param: torch.Tensor, dp_state_is_sharded: bool
) -> bool:
    """Select one DP/CP owner while retaining every unique TP/EP matrix shard."""
    if md_optimizer.pg_collection is None or dp_state_is_sharded:
        return True
    is_expert = getattr(param, "expert_tp", False)
    dp_group = (
        getattr(md_optimizer.pg_collection, "expt_dp", None)
        if is_expert
        else getattr(md_optimizer.pg_collection, "dp_cp", None)
    )
    return dp_group is None or get_pg_rank(dp_group) == 0


def _accumulate_reduced_matrix_batch(
    reduced: torch.Tensor,
    metadata,
    matrix_axis_totals: torch.Tensor,
    matrix_totals: torch.Tensor,
    sparsity_thresholds: tuple[float, ...],
    log_gains: bool,
    log_param_rms: bool,
) -> None:
    """Compute per-matrix metrics after TP has reconstructed each gain axis.

    ``effective-rms`` is computed separately for every matrix. A row/column pair from that same
    matrix produces ``gain-field/rms`` and, for softplus gains, the two gauge metrics. Sparsity
    is the fraction of gain-baked weight values whose magnitude is below each configured
    threshold. The resulting values are summed here and divided by their matrix counts in
    :func:`_append_md_gain_stats`.
    """
    offset = 0
    for family_index, layer, matrix_count, present_axes, is_softplus in metadata:
        values = reduced[offset : offset + matrix_count]
        offset += matrix_count
        scopes = (
            (0, layer + 1)
            if layer is not None and layer + 1 < matrix_axis_totals.size(0)
            else (0,)
        )
        if log_gains:
            for axis_index, axis in enumerate(_GAIN_AXES):
                if axis not in present_axes:
                    continue
                matrix_rms = torch.sqrt(
                    values[:, axis_index, _MATRIX_SQUARE_SUM]
                    / values[:, axis_index, _MATRIX_ELEMENT_COUNT]
                )
                bucket = family_index * len(_GAIN_AXES) + axis_index
                for scope in scopes:
                    matrix_axis_totals[scope, bucket, _MATRIX_RMS_SUM].add_(
                        matrix_rms.sum()
                    )
                    matrix_axis_totals[scope, bucket, _MATRIX_COUNT].add_(matrix_count)

        if sparsity_thresholds:
            weight_element_count = values[:, 0, _WEIGHT_ELEMENT_COUNT]
            for threshold_index in range(len(sparsity_thresholds)):
                sparsity = (
                    values[:, 0, _MATRIX_NEAR_ZERO_START + threshold_index]
                    / weight_element_count
                )
                for scope in scopes:
                    matrix_totals[
                        scope,
                        family_index,
                        _SPARSITY_SUM_START + threshold_index,
                    ].add_(sparsity.sum())
            for scope in scopes:
                matrix_totals[scope, family_index, _SPARSITY_MATRIX_COUNT].add_(
                    matrix_count
                )

        if log_param_rms:
            param_rms = torch.sqrt(
                values[:, 0, _WEIGHT_SQUARE_SUM] / values[:, 0, _WEIGHT_ELEMENT_COUNT]
            )
            for scope in scopes:
                matrix_totals[scope, family_index, _PARAM_RMS_SUM].add_(param_rms.sum())
                matrix_totals[scope, family_index, _PARAM_RMS_COUNT].add_(matrix_count)

        if not log_gains or "row" not in present_axes or "col" not in present_axes:
            continue
        row = values[:, _GAIN_AXES.index("row")]
        col = values[:, _GAIN_AXES.index("col")]
        # RMS of the complete outer-product gain field: RMS(r c^T) = RMS(r) * RMS(c).
        row_rms = torch.sqrt(row[:, _MATRIX_SQUARE_SUM] / row[:, _MATRIX_ELEMENT_COUNT])
        col_rms = torch.sqrt(col[:, _MATRIX_SQUARE_SUM] / col[:, _MATRIX_ELEMENT_COUNT])
        scale = row_rms * col_rms

        for scope in scopes:
            matrix_totals[scope, family_index, _SCALE_SUM].add_(scale.sum())
            matrix_totals[scope, family_index, _PAIR_COUNT].add_(matrix_count)

        if is_softplus:
            row_mean_log = row[:, _MATRIX_LOG_SUM] / row[:, _MATRIX_ELEMENT_COUNT]
            col_mean_log = col[:, _MATRIX_LOG_SUM] / col[:, _MATRIX_ELEMENT_COUNT]
            combined_sum = (row_mean_log + col_mean_log).sum()
            imbalance_sum = (row_mean_log - col_mean_log).sum()
            for scope in scopes:
                matrix_totals[scope, family_index, _COMBINED_LOG_SCALE_SUM].add_(
                    combined_sum
                )
                matrix_totals[scope, family_index, _ROW_COL_IMBALANCE_SUM].add_(
                    imbalance_sum
                )
                matrix_totals[scope, family_index, _SOFTPLUS_PAIR_COUNT].add_(
                    matrix_count
                )


def _accumulate_md_matrix_stats(
    md_optimizers: List["MDDecoupling"],
    dp_state_is_sharded: bool,
    family_indices: Dict[str, int],
    matrix_axis_totals: torch.Tensor,
    matrix_totals: torch.Tensor,
    sparsity_thresholds: tuple[float, ...],
    log_gains: bool,
    log_param_rms: bool,
) -> None:
    """Build the sufficient statistics needed for matrix-level metrics.

    Each record contains gain statistics plus the near-zero and element counts of the gain-baked
    weight. TP all-reduce reconstructs sharded matrices and gain axes before matrix metrics are
    computed. Merged 3D expert parameters retain one record per local expert matrix.
    """
    batches = {}
    for md_optimizer in md_optimizers:
        for param, state in md_optimizer.state.items():
            if not _include_param_in_matrix_stats(
                md_optimizer, param, dp_state_is_sharded
            ):
                continue
            present_axes = [axis for axis in _GAIN_AXES if f"{axis}_gain" in state]
            if not present_axes:
                continue

            is_expert = getattr(param, "expert_tp", False)
            tp_group = None
            if md_optimizer.pg_collection is not None:
                tp_group = (
                    getattr(md_optimizer.pg_collection, "expt_tp", None)
                    if is_expert
                    else getattr(md_optimizer.pg_collection, "tp", None)
                )
            tp_rank = get_pg_rank(tp_group) if tp_group is not None else 0
            matrix_count = param.size(0) if param.ndim == 3 else 1
            matrix_stat_count = _MATRIX_NEAR_ZERO_START + len(sparsity_thresholds)
            record = torch.zeros(
                (matrix_count, len(_GAIN_AXES), matrix_stat_count),
                dtype=torch.float64,
                device=param.device,
            )
            partition_dim = getattr(param, "partition_dim", None)
            if (sparsity_thresholds or log_param_rms) and (
                partition_dim in {0, 1} or tp_rank == 0
            ):
                # MDDecoupling reapplies gains to the parameter at the end of every optimizer
                # step, before logging runs. ``param`` is therefore already the effective weight.
                matrix_weights = param.detach().reshape(matrix_count, -1)
                dtype_info = torch.finfo(matrix_weights.dtype)
                smallest_positive = dtype_info.tiny * dtype_info.eps
                for threshold_index, threshold in enumerate(sparsity_thresholds):
                    # A threshold below the dtype's smallest positive value selects exact zeros.
                    # Handling that case explicitly avoids underflowing the comparison scalar.
                    below_threshold = (
                        matrix_weights.eq(0)
                        if threshold <= smallest_positive
                        else matrix_weights.abs().lt(threshold)
                    ) # NOTE: Needs to create a full sized temporary matrix for abs and then another int8 one for the boolean comparison.
                    record[:, 0, _MATRIX_NEAR_ZERO_START + threshold_index] = (
                        below_threshold.sum(dim=1)
                    )
                if log_param_rms:
                    record[:, 0, _WEIGHT_SQUARE_SUM] = torch.linalg.vector_norm(
                        matrix_weights, dim=1, dtype=torch.float32
                    ).square()
                record[:, 0, _WEIGHT_ELEMENT_COUNT] = matrix_weights.size(1)
            for axis_index, axis in enumerate(_GAIN_AXES):
                raw_gain = state.get(f"{axis}_gain")
                if raw_gain is None or not log_gains:
                    continue
                retained_dim = {"row": 0, "col": 1, "flat": None}[axis]
                gain_is_tp_sharded = (
                    retained_dim is not None
                    and getattr(param, "partition_dim", None) == retained_dim
                )
                if not gain_is_tp_sharded and tp_rank != 0:
                    continue

                effective_gain = md_optimizer._phi(raw_gain).to(dtype=torch.float64)
                matrix_gains = (
                    effective_gain.reshape(matrix_count, -1)
                    if param.ndim == 3
                    else effective_gain.reshape(1, -1)
                )
                record[:, axis_index, _MATRIX_SQUARE_SUM] = matrix_gains.square().sum(
                    dim=1
                )
                record[:, axis_index, _MATRIX_ELEMENT_COUNT] = matrix_gains.size(1)
                if md_optimizer.gain_parametrization == "softplus":
                    matrix_log_gains = torch.nn.functional.softplus(
                        raw_gain.to(dtype=torch.float64)
                    ).reshape(matrix_count, -1)
                    record[:, axis_index, _MATRIX_LOG_SUM] = matrix_log_gains.log().sum(
                        dim=1
                    )

            records, metadata = batches.setdefault(tp_group, ([], []))
            records.append(record)
            family = getattr(param, "md_gain_log_family", "unclassified")
            metadata.append(
                (
                    family_indices.get(family, family_indices["unclassified"]),
                    getattr(param, "md_gain_log_layer", None),
                    matrix_count,
                    present_axes,
                    md_optimizer.gain_parametrization == "softplus",
                )
            )

    pending = []
    for tp_group, (records, metadata) in batches.items():
        reduced = torch.cat(records)
        work = None
        if tp_group is not None and get_pg_size(tp_group) > 1:
            work = torch.distributed.all_reduce(reduced, group=tp_group, async_op=True)
        pending.append((tp_group, reduced, metadata, work))

    for tp_group, reduced, metadata, work in pending:
        if work is not None:
            work.wait()
        if tp_group is not None and get_pg_rank(tp_group) != 0:
            continue
        _accumulate_reduced_matrix_batch(
            reduced,
            metadata,
            matrix_axis_totals,
            matrix_totals,
            sparsity_thresholds,
            log_gains,
            log_param_rms,
        )


def _append_md_gain_stats(
    stats: Dict[str, float],
    totals: torch.Tensor,
    minima: torch.Tensor,
    maxima: torch.Tensor,
    matrix_axis_totals: torch.Tensor,
    matrix_totals: torch.Tensor,
    sparsity_thresholds: tuple[float, ...],
    prefix: str = "muon-md",
) -> None:
    """Turn reduced sufficient statistics into the final scalar metric values.

    ``mean``, ``rms``, saturation, ``min``, and ``max`` describe all gain elements in a bucket.
    ``effective-rms``, combined scale, gauge, and sparsity metrics instead average
    already-computed matrix values, so differently sized matrices receive equal weight.
    """
    for family_index, family in enumerate(_GAIN_FAMILIES):
        for axis_index, axis in enumerate(_GAIN_AXES):
            bucket = family_index * len(_GAIN_AXES) + axis_index
            total, sum_square, count, softplus_count, saturated_count = totals[
                bucket
            ].tolist()
            if count == 0:
                continue
            gain_prefix = f"{prefix}/gains/{family}/{axis}"
            stats[f"{gain_prefix}/mean"] = total / count
            stats[f"{gain_prefix}/rms"] = math.sqrt(sum_square / count)
            matrix_rms_sum, matrix_count = matrix_axis_totals[bucket].tolist()
            stats[f"{gain_prefix}/effective-rms"] = matrix_rms_sum / matrix_count
            stats[f"{gain_prefix}/min"] = minima[bucket].item()
            stats[f"{gain_prefix}/max"] = maxima[bucket].item()
            if softplus_count:
                stats[f"{gain_prefix}/saturated-fraction"] = (
                    saturated_count / softplus_count
                )

        scale_sum, pair_count, combined_sum, imbalance_sum, softplus_pair_count = (
            matrix_totals[family_index, :_SPARSITY_MATRIX_COUNT].tolist()
        )
        if pair_count:
            stats[f"{prefix}/gain-field/{family}/rms"] = scale_sum / pair_count
        if softplus_pair_count:
            stats[f"{prefix}/gauge/{family}/combined-log-scale"] = (
                combined_sum / softplus_pair_count
            )
            stats[f"{prefix}/gauge/{family}/row-col-imbalance"] = (
                imbalance_sum / softplus_pair_count
            )
        sparsity_matrix_count = matrix_totals[
            family_index, _SPARSITY_MATRIX_COUNT
        ].item()
        if sparsity_matrix_count:
            for threshold_index, threshold in enumerate(sparsity_thresholds):
                sparsity_sum = matrix_totals[
                    family_index,
                    _SPARSITY_SUM_START + threshold_index,
                ].item()
                stats[f"{prefix}/sparsity/{family}/fraction-below-{threshold}"] = (
                    sparsity_sum / sparsity_matrix_count
                )
        param_rms_sum = matrix_totals[family_index, _PARAM_RMS_SUM].item()
        param_rms_count = matrix_totals[family_index, _PARAM_RMS_COUNT].item()
        if param_rms_count:
            stats[f"{prefix}/params/{family}/rms"] = param_rms_sum / param_rms_count


@torch.no_grad()
def collect_md_gain_stats(
    optimizer,
    per_layer: bool = False,
    sparsity_thresholds=_DEFAULT_SPARSITY_THRESHOLDS,
    log_gains: bool = True,
    log_sparsity: bool = True,
    log_param_rms: bool = True,
) -> Dict[str, float]:
    """Collect selected global and optionally per-layer Muon-MD statistics.

    Scope 0 holds the global aggregates; scope ``layer + 1`` holds a layer's aggregates. Local
    sufficient statistics are accumulated on-device, all scopes are reduced together, and only
    the final scalar divisions happen on CPU.
    """
    if not log_gains and not log_sparsity and not log_param_rms:
        return {}
    sparsity_thresholds = (
        tuple(dict.fromkeys(float(value) for value in sparsity_thresholds))
        if log_sparsity
        else ()
    )
    if log_sparsity and (
        not sparsity_thresholds
        or any(not math.isfinite(value) or value <= 0 for value in sparsity_thresholds)
    ):
        raise ValueError("Muon-MD sparsity thresholds must be finite positive values")

    # Local import avoids a module cycle: MDDecoupling imports the lightweight family classifier.
    from .md_decoupling import MDDecoupling

    wrapped_optimizers = getattr(optimizer, "chained_optimizers", (optimizer,))
    md_optimizers = [
        getattr(wrapped, "optimizer", wrapped) for wrapped in wrapped_optimizers
    ]
    md_optimizers = [
        wrapped for wrapped in md_optimizers if isinstance(wrapped, MDDecoupling)
    ]
    dp_state_is_sharded = isinstance(optimizer, LayerWiseDistributedOptimizer)
    distributed = (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    )
    if not md_optimizers and not distributed:
        return {}

    bucket_count = len(_GAIN_FAMILIES) * len(_GAIN_AXES)
    gain_params = (
        param
        for md_optimizer in md_optimizers
        for param, state in md_optimizer.state.items()
        if any(f"{axis}_gain" in state for axis in _GAIN_AXES)
    )
    gain_param = next(gain_params, None)
    device = (
        gain_param.device
        if gain_param is not None
        else (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
    )

    layer_count = 0
    if per_layer:
        layer_count = max(
            (
                getattr(param, "md_gain_log_layer", -1) + 1
                for md_optimizer in md_optimizers
                for param in md_optimizer.state
            ),
            default=0,
        )
        if distributed:
            layer_count_tensor = torch.tensor(
                layer_count, dtype=torch.int64, device=device
            )
            torch.distributed.all_reduce(
                layer_count_tensor, op=torch.distributed.ReduceOp.MAX
            )
            layer_count = int(layer_count_tensor.item())
    scope_count = layer_count + 1
    totals = torch.zeros(
        (scope_count, bucket_count, _GAIN_STAT_COUNT),
        dtype=torch.float64,
        device=device,
    )
    minima = torch.full(
        (scope_count, bucket_count), float("inf"), dtype=torch.float64, device=device
    )
    maxima = torch.full_like(minima, float("-inf"))
    matrix_axis_totals = torch.zeros(
        (scope_count, bucket_count, _MATRIX_AXIS_STAT_COUNT),
        dtype=torch.float64,
        device=device,
    )
    matrix_totals = torch.zeros(
        (
            scope_count,
            len(_GAIN_FAMILIES),
            _SPARSITY_SUM_START + len(sparsity_thresholds),
        ),
        dtype=torch.float64,
        device=device,
    )

    family_indices = {name: index for index, name in enumerate(_GAIN_FAMILIES)}
    _accumulate_md_matrix_stats(
        md_optimizers,
        dp_state_is_sharded,
        family_indices,
        matrix_axis_totals,
        matrix_totals,
        sparsity_thresholds,
        log_gains,
        log_param_rms,
    )
    for md_optimizer in md_optimizers:
        for param, state in md_optimizer.state.items():
            if not log_gains:
                continue
            family = getattr(param, "md_gain_log_family", "unclassified")
            family_index = family_indices.get(family, family_indices["unclassified"])
            for axis_index, axis in enumerate(_GAIN_AXES):
                raw_gain = state.get(f"{axis}_gain")
                if raw_gain is None or not _include_gain_in_global_stats(
                    md_optimizer, param, axis, dp_state_is_sharded
                ):
                    continue
                effective_gain = md_optimizer._phi(raw_gain).to(dtype=torch.float64)
                bucket = family_index * len(_GAIN_AXES) + axis_index
                gain_sum = effective_gain.sum()
                gain_square_sum = effective_gain.square().sum()
                gain_count = effective_gain.numel()
                gain_min = effective_gain.min()
                gain_max = effective_gain.max()
                saturated_count = None

                if md_optimizer.gain_parametrization == "softplus":
                    # sigmoid(raw_gain) is d softplus(raw_gain) / d raw_gain. Below 1e-2, the
                    # effective multiplier changes by less than 1% of the raw-gain update.
                    saturated_count = torch.sigmoid(raw_gain).lt(1e-2).sum()
                layer = getattr(param, "md_gain_log_layer", None)
                scopes = (0, layer + 1) if per_layer and layer is not None else (0,)
                for scope in scopes:
                    totals[scope, bucket, _GAIN_SUM].add_(gain_sum)
                    totals[scope, bucket, _GAIN_SQUARE_SUM].add_(gain_square_sum)
                    totals[scope, bucket, _GAIN_ELEMENT_COUNT].add_(gain_count)
                    if saturated_count is not None:
                        totals[scope, bucket, _GAIN_SOFTPLUS_COUNT].add_(gain_count)
                        totals[scope, bucket, _GAIN_SATURATED_COUNT].add_(
                            saturated_count
                        )
                    minima[scope, bucket].copy_(
                        torch.minimum(minima[scope, bucket], gain_min)
                    )
                    maxima[scope, bucket].copy_(
                        torch.maximum(maxima[scope, bucket], gain_max)
                    )

    if distributed:
        works = []
        if log_gains:
            works.extend(
                (
                    torch.distributed.all_reduce(totals, async_op=True),
                    torch.distributed.all_reduce(
                        minima, op=torch.distributed.ReduceOp.MIN, async_op=True
                    ),
                    torch.distributed.all_reduce(
                        maxima, op=torch.distributed.ReduceOp.MAX, async_op=True
                    ),
                    torch.distributed.all_reduce(matrix_axis_totals, async_op=True),
                )
            )
        works.append(torch.distributed.all_reduce(matrix_totals, async_op=True))
        for work in works:
            work.wait()

    totals, minima, maxima, matrix_axis_totals, matrix_totals = (
        tensor.cpu()
        for tensor in (totals, minima, maxima, matrix_axis_totals, matrix_totals)
    )
    stats = {}
    for scope in range(scope_count):
        prefix = "muon-md" if scope == 0 else f"muon-md/layers/{scope - 1}"
        _append_md_gain_stats(
            stats,
            totals[scope],
            minima[scope],
            maxima[scope],
            matrix_axis_totals[scope],
            matrix_totals[scope],
            sparsity_thresholds,
            prefix=prefix,
        )
    return stats
