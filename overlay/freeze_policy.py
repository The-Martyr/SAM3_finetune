"""Explicit, auditable V/T/F/D/M freezing for SAM3 specialist training."""

from collections import defaultdict


POLICIES = {
    "fdm": frozenset("FDM"),
    "tfdm": frozenset("TFDM"),
    "vfdm": frozenset("VFDM"),
    "full": frozenset("VTFDM"),
}


def _group_for_name(name):
    if name.startswith("backbone.vision_backbone."):
        return "V"
    if name.startswith("backbone.language_backbone."):
        return "T"
    if name.startswith("transformer.encoder.") or name.startswith("geometry_encoder."):
        return "F"
    if name.startswith("transformer.decoder.") or name.startswith("dot_prod_scoring."):
        return "D"
    # The semantic segmentation head has a separate objective. Our M ladder is
    # instance-mask supervision; loss_fn_semantic_seg is intentionally null.
    if name.startswith("segmentation_head.semantic_seg_head."):
        return "S"
    if name.startswith("segmentation_head."):
        return "M"
    return None


def apply_freeze_policy(model, policy):
    policy = policy.lower()
    if policy not in POLICIES:
        raise ValueError(f"Unknown freeze policy {policy}; choose {sorted(POLICIES)}")
    enabled = POLICIES[policy]
    totals = defaultdict(int)
    trainable = defaultdict(int)
    unmatched = []
    for name, param in model.named_parameters():
        group = _group_for_name(name)
        if group is None:
            unmatched.append(name)
            continue
        param.requires_grad_(group in enabled)
        totals[group] += param.numel()
        if param.requires_grad:
            trainable[group] += param.numel()
    if unmatched:
        raise RuntimeError("Unclassified SAM3 parameters:\n" + "\n".join(unmatched))
    # requires_grad=False does not change global grad mode during forward. The
    # release ViT fused MLP requires frozen inference under no_grad, and doing so
    # also avoids storing V/T activations while preserving gradients downstream.
    model.backbone.freeze_vision_forward = "V" not in enabled
    model.backbone.freeze_text_forward = "T" not in enabled
    report = {
        "policy": policy,
        "enabled": "".join(sorted(enabled)),
        "total": dict(totals),
        "trainable": dict(trainable),
    }
    print(f"SAM3_FREEZE_AUDIT={report}")
    model._freeze_policy_report = report
    return report
