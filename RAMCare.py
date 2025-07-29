import math

import torch
from torch import nn

from MIMIC3_mTand_txt.mTand_txt_model import MIMIC3_mTand_txt
from RAMCare.RAM_reconstruct import OffsetRAMModule
from RAMCare.classifier import ClassificationHead
from UTDE_MIMIC3.UTDE_model import UTDE_MIMIC3


import torch
import torch.nn as nn
import torch.nn.functional as F

from RAMCare.loss import RESTORE_LOSS


class RAMCare(nn.Module):
    def __init__(self,args,Biobert=None):
        super(RAMCare, self).__init__()

        self.stage = args.stage
        self.ts_embed_dim = args.ts_embed_dim
        self.ns_embed_dim = args.ns_embed_dim

        if "UTDE" in args.ts_model:
            self.TS_MODEL = UTDE_MIMIC3(args)

        if "mTand" in args.ns_model:
            self.NS_MODEL = MIMIC3_mTand_txt(args, Biobert)

        self.TS2NS_Module = OffsetRAMModule(cluster_num=args.ts_cluster_num,
                                     input_hd=args.ts_output_dim,
                                     output_hd=args.ns_embed_dim*2,
                                     method=args.ts2ns_offset_method,
                                     fusion=args.ts2ns_fusion,
                                     num_heads=args.rag_head_num,
                                     sim_type=args.rag_sim_type,
                                     sim_mode=args.ts_sim_mode,
                                     topk_mask=args.ts_topk_mask
                                     )

        self.NS2TS_Module = OffsetRAMModule(cluster_num=args.ns_cluster_num,
                                     input_hd=args.ns_embed_dim*2,
                                     output_hd=args.ts_output_dim,
                                     method=args.ns2ts_offset_method,
                                     fusion=args.ns2ts_fusion,
                                     num_heads=args.rag_head_num,
                                     sim_type=args.rag_sim_type,
                                     sim_mode=args.ns_sim_mode,
                                     topk_mask=args.ns_topk_mask
                                     )

        self.classifier = ClassificationHead(args, args.device)

        self.restore_LOSS = RESTORE_LOSS()


    def forward(self, demogra, ts_data, ts_tt, ts_mask, ts_tau, note_token_id, note_attention_mask, note_tt, note_tau, note_mask, query_ts_tt=None, query_note_tt=None, query_ts_data=None, ts_miss=None, ns_miss=None):
        B, N, T = note_token_id.shape

        # single model
        ts_out, ts_hourly_rep, ts_hd = self.TS_MODEL(demogra, ts_data, ts_tt, ts_mask, ts_tau, note_token_id,
                                                 note_attention_mask, note_tt,
                                                 note_tau, note_mask, query_ts_tt=query_ts_tt, query_note_tt=query_note_tt,
                                                 query_ts_data=query_ts_data)

        ns_out, note_enc_mean, ns_hourly_rep, ns_hd = self.NS_MODEL(demogra, ts_data, ts_tt, ts_mask, ts_tau, note_token_id,
                                                               note_attention_mask, note_tt,
                                                               note_tau, note_mask, query_ts_tt=query_ts_tt,
                                                               query_note_tt=query_note_tt, query_ts_data=query_ts_data)


        # reconstruction
        # hidden_state -> RAM + offset reconstruct
        ns_reconstruct_hd, ns_rag_re, ns_offset_re = self.TS2NS_Module(ts_hd)  # B 2*D
        ts_reconstruct_hd, ts_rag_re, ts_offset_re = self.NS2TS_Module(ns_hd)  # B 2*D

        if self.stage == "pretrain":
            # reconstruct loss
            ts2ns_loss = self.restore_LOSS(ns_reconstruct_hd, ns_hd)
            ns2ts_loss = self.restore_LOSS(ts_reconstruct_hd, ts_hd)

        elif self.stage == "finetune":
            ts_miss_mask = ts_miss.bool()  # [B]
            if ts_miss_mask.any():
                B, L, _ = ts_hourly_rep.shape
                ts_hourly_rep = ts_hourly_rep.clone()
                ts_replace_values = ts_reconstruct_hd[ts_miss_mask]  # [M, 2D]
                ts_replace_expanded = ts_replace_values.unsqueeze(1).expand(-1, L, -1)  # [M, L, 2D]
                ts_hourly_rep[ts_miss_mask] = ts_replace_expanded.to(ts_hourly_rep.dtype)

            ns_miss_mask = ns_miss.bool()  # [B]
            if ns_miss_mask.any():
                B, L, _ = ns_hourly_rep.shape
                ns_hourly_rep = ns_hourly_rep.clone()
                note_enc_mean = note_enc_mean.clone()

                ns_replace_values = ns_reconstruct_hd[ns_miss_mask, :self.ns_embed_dim]  # [B, 2D] -> [M, D]
                ns_replace_expanded = ns_replace_values.unsqueeze(1).expand(-1, L, -1)  # [M, L, D]

                ns_hourly_rep[ns_miss_mask] = ns_replace_expanded.to(ns_hourly_rep.dtype)
                note_enc_mean[ns_miss_mask] = ns_reconstruct_hd[ns_miss_mask, self.ns_embed_dim:].to(note_enc_mean.dtype)

            # reconstruct loss
            ts2ns_loss = self.restore_LOSS(ns_reconstruct_hd, ns_hd)
            ns2ts_loss = self.restore_LOSS(ts_reconstruct_hd, ts_hd)

        # classifier
        out = self.classifier(ts_hourly_rep, ns_hourly_rep, note_enc_mean)  # [B L D] [B L D] [B D]
        others = ts2ns_loss, ns2ts_loss

        return out, others




