import logging
from typing import List, Optional, Tuple

import math
import torch
import torch.nn as nn
from torch.nn import functional as F


from transformers import GptOssConfig
from transformers.cache_utils import Cache
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssMLP as GptOssMLPBase, GptOssTopKRouter as GptOssTopKRouterBase, GptOssExperts
from transformers.integrations.hub_kernels import use_kernel_forward_from_hub

from .moe_utils import (compute_routing_scores_for_aux_loss, 
                        switch_load_balancing_loss_func,
                        save_to_aux_losses_tracker,
                        MoEAuxLossAutoScaler)

from .base import Eagle3DraftModel
from .llama3_eagle import LlamaForCausalLMEagle3

logger = logging.getLogger(__name__)

class GptOssTopKRouter(GptOssTopKRouterBase):
    def __init__(self, config):
        super().__init__(config)
        self.is_aux_loss_enabled = True
        self.moe_aux_loss_coeff = config.moe_aux_loss_coeff
        self.routing_type = config.moe_router_load_balancing_type
        self.num_hidden_layers = config.num_hidden_layers
        self.layer_number = 0
        self.step = 0
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)
        print(f"init router weight & bias")


        # Initialize global tokens per expert for global aux loss
        if self.get_aux_loss_coeff("global_aux_loss") > 0:
            self.register_buffer(
                'global_tokens_per_expert',
                torch.zeros(
                    self.num_experts,
                    dtype=torch.float32,
                ),
                persistent=False,
            )
            self.register_buffer(
                'ga_steps',
                torch.tensor(0, dtype=torch.float32),
                persistent=False,
            )
        else:
            self.global_tokens_per_expert = None
            self.ga_steps = None
    
    def forward(self, hidden_states):
        bsz, seq_length = hidden_states.shape[:2]

        hidden_states = hidden_states.reshape(-1, self.hidden_dim) # (tokens, hidden_dim)
        router_logits = F.linear(hidden_states, self.weight, self.bias)  # (seq_len, num_experts)

        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)  # (seq_len, top_k)
        router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
        router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)

        if self.is_aux_loss_enabled:
            routing_map_for_aux_loss, scores_for_aux_loss = compute_routing_scores_for_aux_loss(router_logits, self.top_k)
            router_scores = self._apply_aux_loss(router_scores, scores_for_aux_loss, routing_map_for_aux_loss)
            # router_scores = self._apply_seq_aux_loss(
            #     router_scores, scores_for_aux_loss, routing_map_for_aux_loss, seq_length, bsz
            # )
            # router_scores = self._apply_global_aux_loss(
            #     router_scores, scores_for_aux_loss, routing_map_for_aux_loss
            # )

        return router_scores, router_indices

    def _apply_aux_loss(
        self, probs: torch.Tensor, scores_for_aux_loss: torch.Tensor, routing_map: torch.Tensor
    ):
        """Apply the auxiliary loss for the given scores and routing map."""
        aux_loss_coeff = self.get_aux_loss_coeff("aux_loss")
        if aux_loss_coeff == 0:
            return probs
        tokens_per_expert = routing_map.sum(dim=0)
        num_tokens = routing_map.shape[0]
        total_num_tokens = num_tokens * 1

        aux_loss = switch_load_balancing_loss_func(
            probs=scores_for_aux_loss,
            tokens_per_expert=tokens_per_expert,
            total_num_tokens=total_num_tokens,
            topk=self.top_k,
            num_experts=self.num_experts,
            moe_aux_loss_coeff=aux_loss_coeff,
        )
        probs = self.attach_and_log_load_balancing_loss(
            probs, aux_loss_coeff, aux_loss, "load_balancing_loss"
        )
        return probs

    def _apply_seq_aux_loss(
        self,
        probs: torch.Tensor,
        scores_for_aux_loss: torch.Tensor,
        routing_map: torch.Tensor,
        seq_length: int,
        bsz: int,
    ):
        """Apply the sequence-level auxiliary loss for the given scores and routing map.

        To calculate the sequence-level aux loss, we reshape the batch_size dimension to
        experts dimension. The resulted loss by switch_load_balancing_loss_func is equal
        to the sum of aux loss for each sequence in the batch. And then we divide the aux
        loss by the batch size to get averaged aux loss.
        """
        seq_aux_loss_coeff = self.get_aux_loss_coeff("seq_aux_loss")
        if seq_aux_loss_coeff == 0:
            return probs

        scores_for_aux_loss = scores_for_aux_loss.reshape(seq_length, -1)
        tokens_per_expert = routing_map.reshape(seq_length, -1).sum(dim=0)

        total_num_tokens = seq_length * 1

        aux_loss = (
            switch_load_balancing_loss_func(
                probs=scores_for_aux_loss,
                tokens_per_expert=tokens_per_expert,
                total_num_tokens=total_num_tokens,
                topk=self.top_k,
                num_experts=self.num_experts,
                moe_aux_loss_coeff=seq_aux_loss_coeff,
            )
            / bsz
        )
        probs = self.attach_and_log_load_balancing_loss(
            probs, seq_aux_loss_coeff, aux_loss, "seq_load_balancing_loss"
        )
        return probs

    def _apply_global_aux_loss(
        self, probs: torch.Tensor, scores_for_aux_loss: torch.Tensor, routing_map: torch.Tensor
    ):
        """Apply the global auxiliary loss for the given scores and routing map."""
        global_aux_loss_coeff = self.get_aux_loss_coeff("global_aux_loss")
        if global_aux_loss_coeff == 0:
            return probs

        tokens_per_expert = routing_map.sum(dim=0)

        self.global_tokens_per_expert += tokens_per_expert
        self.ga_steps += 1
        averated_tokens_per_expert = self.global_tokens_per_expert / self.ga_steps

        num_tokens = scores_for_aux_loss.shape[0]
        total_num_tokens = num_tokens * 1 

        global_aux_loss = switch_load_balancing_loss_func(
            probs=scores_for_aux_loss,
            tokens_per_expert=averated_tokens_per_expert,
            total_num_tokens=total_num_tokens,
            topk=self.top_k,
            num_experts=self.num_experts,
            moe_aux_loss_coeff=global_aux_loss_coeff,
        )
        probs = self.attach_and_log_load_balancing_loss(
            probs,
            global_aux_loss_coeff,
            global_aux_loss,
            "global_load_balancing_loss",
        )
        return probs
    
    def get_aux_loss_coeff(self, aux_loss_type: str) -> float:
        """Return the aux loss coeff for the given auxiliary loss type.
        If the auxiliary loss type is not found, return 0.0.
        """
        if isinstance(self.routing_type, str):
            if self.routing_type == aux_loss_type:
                return self.moe_aux_loss_coeff
        if isinstance(self.routing_type, list):
            try:
                idx = self.routing_type.index(aux_loss_type)
                return self.moe_aux_loss_coeff[idx]
            except ValueError:
                return 0.0
        return 0.0
    
    def attach_and_log_load_balancing_loss(
        self,
        activation: torch.Tensor,
        aux_loss_coeff: float,
        aux_loss: torch.Tensor,
        aux_loss_name: str,
    ):
        """Attach aux loss function to activation and add to logging."""
        num_layers = self.num_hidden_layers
        save_to_aux_losses_tracker(
            aux_loss_name,
            aux_loss / aux_loss_coeff,
            self.layer_number,
            num_layers,
            self.step,
        )
        self.step += 1
        activation = MoEAuxLossAutoScaler.apply(activation, aux_loss)
        return activation

@use_kernel_forward_from_hub("MegaBlocksMoeMLP")
class GptOssMLP(GptOssMLPBase):
    def __init__(self, config):
        super().__init__(config)
        self.router = GptOssTopKRouter(config)

class GptOssForCausalLMEagle3(LlamaForCausalLMEagle3):

    config_class = GptOssConfig

    def __init__(self, config, quant_config=None, attention_backend="sdpa") -> None:
        print(f"{config=}")
        super().__init__(config, attention_backend=attention_backend)
        self.midlayer.mlp = GptOssMLP(config)


__all__ = ["GptOssForCausalLMEagle3"]