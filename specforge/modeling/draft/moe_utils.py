import torch
from specforge.tracker import get_tracker, get_step_counter

def compute_routing_scores_for_aux_loss(
    logits: torch.Tensor, topk: int
):
    """Compute routing scores based on the score function.

    Args:
        logits (torch.Tensor): The logits tensor after gating, shape: [num_tokens, num_experts].

    Returns:
        torch.Tensor: The normalized routing scores.
    """
    scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
    _, top_indices = torch.topk(scores, k=topk, dim=1)
    routing_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()
    return routing_map, scores

def switch_load_balancing_loss_func(
    probs: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    total_num_tokens: int,
    topk: int,
    num_experts: int,
    moe_aux_loss_coeff: float,
):
    """Calculate the auxiliary loss for load balancing.
    Refer to the Switch Transformer (https://arxiv.org/abs/2101.03961)
    and Global Load Balancing Loss(https://arxiv.org/abs/2501.11873) for details.

    ### Detailed explanation of the auxiliary loss #######

    The formula for the auxiliary loss is:
        loss = E * Σ_{i=1}^{E} (f_i * P_i)
    where:
        f_i = 1 / (T * topk) * Σ_{x∈B} routing_map(x, i)
             (fraction of tokens dispatched to expert i)
        P_i = 1 / T * Σ_{x∈B} probs(x, i)
             (averaged router probability allocated for expert i)
        E is the number of experts
        T is the total number of tokens in the batch B

    For distributed training with sequence or context parallelism, each rank can
    process a subset of the batch.
        loss = E * Σ_{i=1}^{E} (f_i * Σ_{j=1}^{N} P_ij)
             = E * Σ_{i=1}^{E} Σ_{j=1}^{N} (f_i * P_ij)
             = Σ_{j=1}^{N} E * (Σ_{i=1}^{E} f_i * P_ij)

    where:
        f_i = 1 / (T * topk) * Σ_{x∈B} routing_map(x, i)
             (fraction of tokens dispatched to expert i in the global batch)
        P_ij = 1 / T * Σ_{x∈B_j} probs(x, i)
              (averaged router probability allocated for expert i in local batch of the j-th rank)
        N is the number of ranks
        B_j is the batch of tokens in the j-th rank
        T is the total number of tokens in the global batch B

    Note:
    To calculate the auxiliary loss at different levels (micro-batch or global batch):
    - probs: Should always be from the local batch being processed
    - tokens_per_expert: Should represent token counts at the desired level
      (either micro-batch or global batch)
    - total_num_tokens: Should match the total token count at the same level as tokens_per_expert

    #########################################################

    Args:
        probs (torch.Tensor): Softmax probabilities output by the router for each token.
                              Shape in [num_tokens, num_experts].
        tokens_per_expert (torch.Tensor): Number of tokens assigned to each expert in the batch.
                                          Shape in [num_experts]
        total_num_tokens (int): Total number of tokens in the batch.
        topk (int): The number of experts selected for each token.
        num_experts (int): The number of experts.
        moe_aux_loss_coeff (float): The coefficient for the auxiliary loss.
    Returns:
        torch.Tensor: The auxiliary loss for load balancing.
    """
    aggregated_probs_per_expert = probs.sum(dim=0)
    aux_loss = torch.sum(aggregated_probs_per_expert * tokens_per_expert) * (
        num_experts * moe_aux_loss_coeff / (topk * total_num_tokens * total_num_tokens)
    )
    return aux_loss

def save_to_aux_losses_tracker(
    name: str,
    loss: torch.Tensor,
    layer_number: int,
    num_layers: int,
    step: int = None,
):
    """Save the auxiliary loss for logging.
    Args:
        name (str): The name of the loss.
        loss (torch.Tensor): The loss tensor.
        layer_number (int): Layer index of the loss.
        num_layers (int): The number of total layers.
        step (int, optional): Step number for logging. If None, will use global step counter.
    """
    # Skip aux loss logging if layer_number is None.
    if layer_number is None:
        print(f"[DEBUG] Skipping aux loss logging because layer_number is None")
        return

    # Create log dict with the aux loss
    log_dict = {
        f"aux/{name}": loss.detach().item()
    }
        
    # Log to tracker·
    # ignore step for now, use get_step_counter 
    get_tracker().log(log_dict, step=get_step_counter(), commit=False)

class MoEAuxLossAutoScaler(torch.autograd.Function):
    """An AutoScaler that triggers the backward pass and scales the grad for auxiliary loss."""

    main_loss_backward_scale: torch.Tensor = None

    @staticmethod
    def forward(ctx, output: torch.Tensor, aux_loss: torch.Tensor):
        """Preserve the aux_loss by storing it in the context to avoid garbage collection.

        Args:
            output (torch.Tensor): The output tensor.
            aux_loss (torch.Tensor): The auxiliary loss tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        ctx.save_for_backward(aux_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Compute and scale the gradient for auxiliary loss..

        Args:
            grad_output (torch.Tensor): The gradient of the output.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The gradient of the output, scaled auxiliary loss
                                               gradient.
        """
        (aux_loss,) = ctx.saved_tensors
        if MoEAuxLossAutoScaler.main_loss_backward_scale is None:
            MoEAuxLossAutoScaler.main_loss_backward_scale = torch.tensor(
                1.0, device=aux_loss.device
            )
        aux_loss_backward_scale = MoEAuxLossAutoScaler.main_loss_backward_scale
        scaled_aux_loss_grad = torch.ones_like(aux_loss) * aux_loss_backward_scale
        return grad_output, scaled_aux_loss_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor):
        """set the scale of the aux loss.

        Args:
            scale (torch.Tensor): The scale value to set. Please ensure that the scale passed in
                                  matches the scale of the main_loss.
        """
        if MoEAuxLossAutoScaler.main_loss_backward_scale is None:
            MoEAuxLossAutoScaler.main_loss_backward_scale = scale
        else:
            MoEAuxLossAutoScaler.main_loss_backward_scale.copy_(scale)
