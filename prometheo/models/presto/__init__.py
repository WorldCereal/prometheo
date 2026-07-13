# These optimizer utilities are model-agnostic and now live in
# prometheo.finetune; they are re-exported here for backward compatibility.
from prometheo.finetune import (
    adjust_learning_rate,
    param_groups_lrd,
    param_groups_weight_decay,
)
from prometheo.finetune import (
    get_layer_id_for_finetuning as get_layer_id_for_rest_finetuning,
)

__all__ = [
    "adjust_learning_rate",
    "get_layer_id_for_rest_finetuning",
    "param_groups_lrd",
    "param_groups_weight_decay",
]
