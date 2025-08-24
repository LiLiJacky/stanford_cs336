import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)
    
class MoE(nn.Module):
    def __init__(self, input_dim, num_experts, top_k, expert_capacity, hidden_dim, output_dim):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_capacity = expert_capacity

        self.gate = nn.Linear(input_dim, hidden_dim)

        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)
        ])

    def forward(self, x):
        batch_size, input_dim = x.shape
        device = x.device

        logits = self.gate(x)
        probs = torch.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, self.top_k, dim=-1)

        if self.training:
            # 重要性损失（专家利用率均衡）
            importance = probs.sum(0)
            importance_loss = torch.var(importance) / (self.num_experts ** 2)

            # 负载均衡损失（样本分配均衡）
            mask = torch.zeros_like(probs, dtype=torch.bool)
            mask.scatter_(1, topk_indices, True)
            routing_probs = probs * mask
            expert_usage = mask.float().mean(0)
            routing_wieghts = routing_probs.mean(0)
            load_balance_loss = self.num_experts * (expert_usage * routing_wieghts).sum()

            aux_loss = importance_loss + load_balance_loss
        else:
            aux_loss = 0


        flat_indices = topk_indices.view(-1)
        flat_probs = topk_probs.view(-1)
        sample_indices = torch.arange(batch_size, device=device)[:, None]
        sample_indices = sample_indices.expand(-1, self.top_k).flatten()

        outputs = torch.zeros(batch_size, self.experts[0].net[-1].out_features, device=device)

        for expert_idx in range(self.num_experts):
            expert_mask = flat_indices == expert_idx
            expert_samples = sample_indices[expert_mask]
            expert_weights = flat_probs[expert_mask]

            if len(expert_samples) > self.expert_capacity:
                expert_samples = expert_samples[:self.expert_capacity]
                expert_weights = expert_weights[:self.expert_capacity]
            
            if len(expert_samples) == 0:
                continue
                
            expert_input = x[expert_samples]
            expert_output = self.experts[expert_idx](expert_input)
            weighted_output = expert_output * expert_weights.unsqueeze(-1)

            outputs.index_add_(0, expert_samples, weighted_output)
        
        return outputs, aux_loss

if __name__ == '__main__':
    input_dim = 128
    output_dim = 256
    hidden_dim = 512
    num_experts = 8
    top_k = 2
    expert_capacity = 32
    batch_size = 64

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    x = torch.rand(batch_size, input_dim).to(device)
    moe = MoE(input_dim, num_experts, top_k, expert_capacity, hidden_dim, output_dim).to(device)

    for _ in range(1000):
        moe.train()
        output, loss = moe(x)
        print(f"Training output shape: {output.shape}")
        print(f"Auxiliary loss: {loss.item(): .4f}")

    print("=" * 80)

    moe.eval()
    output, _ = moe(x)
    print(f"eval output shape: {output.shape}")