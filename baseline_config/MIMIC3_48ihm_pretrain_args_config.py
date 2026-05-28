import argparse


def MIMIC3_48ihm_pretrain_parse_args():
    parser = argparse.ArgumentParser(description="argparse")

    parser.add_argument("--exp_name", type=str, default="MIMIC3_48ihm_128_256_euc_12_0.7_moe_all_14_0.7_moegate_all_0621")
    parser.add_argument("--stage", type=str, default="pretrain", choices=['pretrain', 'finetune'])
    parser.add_argument("--data_percent", type=str, default="full", choices=['full'])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", type=str, default='48ihm')

    parser.add_argument("--mode", type=str, default='train', choices=["train", "eval"])  # train eval
    parser.add_argument("--tensorboard_dir", type=str, default='./tensorboard/')
    parser.add_argument("--save_dir", type=str, default='./save/')
    parser.add_argument("--log_dir", type=str, default='./log/')

    parser.add_argument('--init_method', default='tcp://localhost:18891', help="init-method")
    parser.add_argument('-g', '--gpuid', default=2, type=int, help="which gpu to use")  # 第几号卡
    parser.add_argument('-r', '--rank', default=0, type=int, help='rank of current process')  # 第几个进程
    parser.add_argument('-w', '--world_size', default=1, type=int, help="world size")  # 一共几张卡

    # data
    parser.add_argument("--var_dim", type=int, default=17)
    parser.add_argument("--invar_dim", type=int, default=2)
    parser.add_argument("--metadata", type=str, default='./data/metadata.json')

    parser.add_argument("--ihm_train_data", type=str, default='./data/MIMIC3/merge/train_normed_')
    parser.add_argument("--ihm_val_data", type=str, default='./data/MIMIC3/merge/val_normed_')
    parser.add_argument("--ihm_test_data", type=str, default='./data/MIMIC3/merge/test_normed_')

    parser.add_argument("--data_modality", type=str, default="all", choices=['ts', 'ns', 'all'])

    # ts data
    parser.add_argument("--ts_max_len", type=int, default=220)  # 220
    parser.add_argument("--impute", default=True)
    parser.add_argument("--conti_impute_type", type=str, default="backward", choices=["linear", "backward", "forward"])  # TODO：插值方法。backward 后面的时刻用前面的有效值插值；forward 前面的时刻用后面的有效值插值

    # note data
    parser.add_argument("--notes_order", type=str, default='Last')
    parser.add_argument("--num_of_notes", type=int, default=5)  # 10
    parser.add_argument("--model_name", type=str, default="bioLongformer")
    parser.add_argument("--max_length", type=int, default=512)  # bioLongformer: 512 or 1024; BioBert: 256 or 512

    parser.add_argument("--mtand_steps", type=int, default=1)

    # optim
    parser.add_argument("--bert_opt", type=str, default="AdamW")
    parser.add_argument("--modal_opt", type=str, default="AdamW")  # single model
    parser.add_argument("--rag_opt", type=str, default="AdamW")
    parser.add_argument("--other_opt", type=str, default="AdamW")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--bert_lr", type=float, default=0.00004)
    parser.add_argument("--modal_lr", type=str, default=0.0001)
    parser.add_argument("--rag_lr", type=float, default=0.001)
    parser.add_argument("--other_lr", type=float, default=0.0004)

    # sche
    parser.add_argument("--bert_sche", type=str, default="linear_with_warmup")
    parser.add_argument("--modal_sche", type=str, default="None")
    parser.add_argument("--rag_sche", type=str, default="None")
    parser.add_argument("--other_sche", type=str, default="None")
    parser.add_argument("--warm_up", type=float, default=0.08)  # 0.08 * total_steps

    # train
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--before_update_bert_epochs", type=int, default=0)  # TODO：bert更新策略 前几个epoch不更新bert
    parser.add_argument("--num_update_bert_epochs", type=int, default=2)  # 间隔几个epoch更新一次bert
    parser.add_argument("--bertcount", type=int, default=3)  # 最多更新几次bert
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)  # 梯度累加 累计多少个step之后更新一次参数

    # eval
    parser.add_argument("--eval", type=bool, default=False)  # TODO: False:不包含eval
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=15)

    # stage: pretrain
    parser.add_argument("--pretrain_model", type=str, default="")

    # test & analysis
    parser.add_argument("--full_model", type=str, default="")
    parser.add_argument("--analysis_data", type=str, default="./data/MIMIC3/analysis/")

    # model: NS modality
    parser.add_argument("--ns_model", type=str, default="mTand_256")
    parser.add_argument("--ns_pretrain_model", type=str, default="./save/48ihm/mTand_txt_MIMIC3_48ihm_dim256_all_0525_seed_42_epoch_50_batch_3_bioLongformer512_emb_dim_256/3.pth")
    parser.add_argument("--ns_embed_dim", type=int, default=256)
    parser.add_argument("--ns_mtand_value_transfer", type=bool, default=False)

    # model: MITS modality
    parser.add_argument("--ts_model", type=str, default="UTDE_128")
    parser.add_argument("--ts_pretrain_model", type=str, default="./save/48ihm/UTDE_MIMIC3_48ihm_all_128_varLoss_output128_0523_seed_42_epoch_50_batch_64_bioLongformer512_emb_dim_128/15.pth")
    parser.add_argument("--ts_embed_dim", type=int, default=128)
    parser.add_argument("--ts_output_dim", type=int, default=128)
    parser.add_argument("--mixup_level", type=str, default="batch_seq", choices=['batch', 'batch_seq', 'batch_seq_feature'])  # impute和mtand
    parser.add_argument("--ts_mtand_value_transfer", type=bool, default=False)

    # RAG
    parser.add_argument("--rag_sim_type", type=str, default="euclidean", choices=['dot', 'cosine', 'euclidean'])
    parser.add_argument("--rag_head_num", type=int, default=1)  # 因为是预载的，具有完整语义，多头切分或许不好

    # model RAG: TS -> NS
    parser.add_argument("--ts2ns_rag", type=str, default="./data/MIMIC3/analysis/128_256/rag_128_256_co-kmeans_")
    parser.add_argument("--ts_cluster_num", type=int, default=11)
    parser.add_argument("--ts_lambda_align", type=float, default=0.7)
    parser.add_argument("--ts_sim_mode", type=str, default="all_sim", choices=['all_sim', 'topk_sim'])
    parser.add_argument("--ts_topk_mask", type=int, default=3)
    parser.add_argument("--ts2ns_offset_method", type=str, default='gate_moe', choices=['mlp', 'concat_sim', 'film', 'moe', 'gate_moe', 'topk_moe', 'subspace', 'bayes'])
    parser.add_argument("--ts2ns_fusion", type=str, default='add', choices=['add', 'gate'])

    # model RAG: NS -> TS
    parser.add_argument("--ns2ts_rag", type=str, default="./data/MIMIC3/analysis/128_256/rag_128_256_co-kmeans_")
    parser.add_argument("--ns_cluster_num", type=int, default=14)
    parser.add_argument("--ns_lambda_align", type=float, default=0.7)
    parser.add_argument("--ns_sim_mode", type=str, default="all_sim", choices=['all_sim', 'topk_sim'])
    parser.add_argument("--ns_topk_mask", type=int, default=5)
    parser.add_argument("--ns2ts_offset_method", type=str, default='gate_moe', choices=['mlp', 'concat_sim', 'film', 'moe', 'gate_moe', 'topk_moe', 'subspace', 'bayes'])
    parser.add_argument("--ns2ts_fusion", type=str, default='add', choices=['add', 'gate'])

    parser.add_argument("--dropout", type=float, default=0.1)

    # classifier
    parser.add_argument("--classifier_nhead", type=int, default=4)
    parser.add_argument("--classifier_layers", type=int, default=1)  # TODO: 2

    # loss
    parser.add_argument("--lambda_ts2ns", type=float, default=50.0)  # TODO:
    parser.add_argument("--lambda_ns2ts", type=float, default=50.0)  # TODO:

    return parser.parse_args()


