import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RAMModule(nn.Module):
    def __init__(self, cluster_num, input_hd, output_hd, num_heads=4, sim_type='dot', sim_mode='all_sim', topk_mask=3):
        super(RAMModule, self).__init__()

        assert sim_type in ['dot', 'cosine', 'euclidean']
        assert sim_mode in ['all_sim', 'topk_sim']
        assert input_hd % num_heads == 0 and output_hd % num_heads == 0, \
            "input_hd and output_hd must be divisible by num_heads"

        self.cluster_num = cluster_num
        self.sim_type = sim_type
        self.sim_mode = sim_mode
        self.topk_mask = topk_mask
        self.num_heads = num_heads
        self.head_dim_in = input_hd // num_heads
        self.head_dim_out = output_hd // num_heads
        self.temperature = math.sqrt(self.head_dim_in)

        self.K = nn.Parameter(torch.randn(num_heads, cluster_num, self.head_dim_in))
        self.V = nn.Parameter(torch.randn(num_heads, cluster_num, self.head_dim_out))

    def compute_similarity(self, Q_h, K_h):
        if self.sim_type == 'dot':
            sim = torch.einsum('bhd,hcd->bhc', Q_h, K_h) / self.temperature
        elif self.sim_type == 'cosine':
            Q_norm = F.normalize(Q_h, dim=-1)
            K_norm = F.normalize(K_h, dim=-1)
            sim = torch.einsum('bhd,hcd->bhc', Q_norm, K_norm) / self.temperature
        elif self.sim_type == 'euclidean':
            Q_exp = Q_h.unsqueeze(2)
            K_exp = K_h.unsqueeze(0)
            dist_sq = torch.sum((Q_exp - K_exp) ** 2, dim=-1)
            sim = -dist_sq / self.temperature
        return sim

    def apply_topk_mask(self, sim):
        if self.sim_mode == 'all_sim':
            return sim
        else:
            topk_val, topk_idx = torch.topk(sim, k=self.topk_mask, dim=-1)
            mask = torch.zeros_like(sim)
            mask.scatter_(-1, topk_idx, 1.0)
            sim = sim.masked_fill(mask == 0, float('-inf'))
            return sim

    def forward(self, Q):
        B = Q.size(0)
        Q_multi = Q.view(B, self.num_heads, self.head_dim_in)
        sim = self.compute_similarity(Q_multi, self.K)

        sim_masked = self.apply_topk_mask(sim)

        attn_weights = F.softmax(sim_masked, dim=-1)  # (B, H, C)
        rag_heads = torch.einsum('bhc,hcd->bhd', attn_weights, self.V)
        rag_reconstruct = rag_heads.contiguous().view(B, -1)

        # === gradient stop for offset path by .detach() ===
        attn_weights_detached = attn_weights.detach()
        offset_heads = torch.einsum('bhc,hcd->bhd', attn_weights_detached, self.K.detach())
        offset_input = offset_heads.contiguous().view(B, -1)
        sim_vector = attn_weights_detached.mean(dim=1)

        return rag_reconstruct, offset_input, sim_vector

    def init_KV(self, K_init, V_init):
        assert K_init.shape == (self.cluster_num, self.num_heads * self.head_dim_in), \
            f"Expected K_init shape ({self.cluster_num}, {self.num_heads * self.head_dim_in})"
        assert V_init.shape == (self.cluster_num, self.num_heads * self.head_dim_out), \
            f"Expected V_init shape ({self.cluster_num}, {self.num_heads * self.head_dim_out})"

        K_init_split = K_init.view(self.cluster_num, self.num_heads, self.head_dim_in).permute(1, 0, 2).contiguous()
        V_init_split = V_init.view(self.cluster_num, self.num_heads, self.head_dim_out).permute(1, 0, 2).contiguous()

        with torch.no_grad():
            self.K.copy_(K_init_split.to(self.K.device))
            self.V.copy_(V_init_split.to(self.V.device))



class OffsetRAMModule(nn.Module):
    def __init__(self, cluster_num, input_hd, output_hd, fusion='add', num_heads=4, sim_type='dot', method='gate_moe', sim_mode="all_sim", topk_mask=3, expert_num=None):
        super(OffsetRAMModule, self).__init__()
        self.rag_module = RAMModule(cluster_num, input_hd, output_hd, num_heads=num_heads, sim_type=sim_type, sim_mode=sim_mode, topk_mask=topk_mask)
        self.method = method
        self.fusion = fusion
        self.output_hd = output_hd
        self.cluster_num = cluster_num
        if expert_num is not None:
            self.expert_num = expert_num
        else:
            self.expert_num = cluster_num

        self.offset_mlp = nn.Sequential(
            nn.Linear(input_hd, output_hd),
            nn.ReLU(),
            nn.Linear(output_hd, output_hd)
        )

        if method in ['moe', 'gate_moe']:
            self.expert_mlps = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_hd, output_hd),
                    nn.ReLU(),
                    nn.Linear(output_hd, output_hd)
                ) for _ in range(self.expert_num)
            ])

        if method == 'gate_moe':
            self.gate_net = nn.Sequential(
                nn.Linear(input_hd + cluster_num, self.expert_num)
            )

    def forward(self, Q):
        rag_reconstruct, offset_input, sim_vector = self.rag_module(Q)
        offset = offset_input - Q

        if self.method == 'gate_moe':
            gate_input = torch.cat([Q, sim_vector], dim=-1)
            routing_weights = F.softmax(self.gate_net(gate_input), dim=-1)
            expert_outputs = torch.stack([expert(offset) for expert in self.expert_mlps], dim=1)
            offset_reconstruct = (routing_weights.unsqueeze(-1) * expert_outputs).sum(dim=1)

        output = rag_reconstruct + offset_reconstruct

        return output, rag_reconstruct, offset_reconstruct

