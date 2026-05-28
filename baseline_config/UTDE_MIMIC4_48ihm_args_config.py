import argparse


def UTDE_MIMIC4_ihm_parse_args():
    parser = argparse.ArgumentParser(description="argparse")

    parser.add_argument("--exp_name", type=str, default="UTDE_MIMIC4_ihm_all_128_varLoss_output_128_attn_1009")
    parser.add_argument("--stage", type=str, default="single_modal")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--task", type=str, default='48ihm')  # 48ihm 24pheno
    parser.add_argument("--mode", type=str, default='train', choices=["train", "eval"])  # train eval
    parser.add_argument("--fp16", type=bool, default=True)
    parser.add_argument("--tensorboard_dir", type=str, default='./tensorboard/')
    parser.add_argument("--save_dir", type=str, default='./save/')
    parser.add_argument("--log_dir", type=str, default='./log/')

    # 多卡训练
    parser.add_argument('-g', '--gpuid', default=0, type=int, help="which gpu to use")  # 第几号卡

    # data
    parser.add_argument("--var_dim", type=int, default=17)
    parser.add_argument("--invar_dim", type=int, default=2)
    parser.add_argument("--cate_dim", type=int, default=0)  # TODO: cate new
    parser.add_argument("--cate_type", type=str, default='4_6_13_5')  # TODO: cate new
    parser.add_argument("--metadata", type=str, default='./data/metadata.json')

    parser.add_argument("--ihm_train_data", type=str, default='./data/MIMIC4/merge/train_')
    parser.add_argument("--ihm_val_data", type=str, default='./data/MIMIC4/merge/val_')
    parser.add_argument("--ihm_test_data", type=str, default='./data/MIMIC4/merge/test_')

    parser.add_argument("--data_modality", type=str, default="ts", choices=['ts', 'ns', 'all'])

    # ts data
    parser.add_argument("--ts_max_len", type=int, default=220)  # 220
    parser.add_argument("--impute", default=True)
    parser.add_argument("--conti_impute_type", type=str, default="backward", choices=["linear", "backward",
                                                                                    "forward"])  # TODO：插值方法。backward 后面的时刻用前面的有效值插值；forward 前面的时刻用后面的有效值插值
    parser.add_argument("--cate_impute_type", type=str, default="backward", choices=["linear", "backward",
                                                                                     "forward"])  # TODO：插值方法。backward 后面的时刻用前面的有效值插值；forward 前面的时刻用后面的有效值插值

    # note data
    parser.add_argument("--notes_order", type=str, default='Last')
    parser.add_argument("--num_of_notes", type=int, default=1)  # 20
    parser.add_argument("--model_name", type=str, default="bioLongformer")
    parser.add_argument("--max_length", type=int, default=512)  # bioLongformer: 512 or 1024; BioBert: 256 or 512

    # optim
    parser.add_argument("--opt", type=str, default="AdamW")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=0.0004)

    parser.add_argument("--variance_loss", type=bool, default=True)  # TODO: 是否包含variance_maximization_loss，用于分散化hidden state表征
    parser.add_argument("--variance_loss_lambda", type=float, default=10.0)

    # sche
    parser.add_argument("--other_sche", type=str, default="None")
    parser.add_argument("--warm_up", type=float, default=0.08)  # 0.08 * total_steps

    # train
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument("--train_batch_size", type=int, default=64)  # 32
    parser.add_argument("--modeltype", type=str, default='TS_Text')
    parser.add_argument("--before_update_bert_epochs", type=int, default=0)  # TODO：bert更新策略 前几个epoch不更新bert
    parser.add_argument("--num_update_bert_epochs", type=int, default=2)  # 间隔几个epoch更新一次bert
    parser.add_argument("--bertcount", type=int, default=3)  # 最多更新几次bert
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)  # 梯度累加 累计多少个step之后更新一次参数

    # eval
    parser.add_argument("--eval", type=bool, default=True)  # TODO: False:不包含eval
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=15)

    # test & analysis
    parser.add_argument("--full_model", type=str, default="./save/48ihm/UTDE_MIMIC4_ihm_all_128_varLoss_output128_0626_seed_42_epoch_50_batch_64_bioLongformer512_emb_dim_128/20.pth")
    parser.add_argument("--analysis_data", type=str, default="./data/MIMIC4/analysis/")

    # model
    parser.add_argument("--ts_embed_dim", type=int, default=128)
    parser.add_argument("--ts_output_dim", type=int, default=128)
    parser.add_argument("--remove_rep", type=str, default="None")  # "None" 'time' 'type' 'density' 'invar'
    parser.add_argument("--mixup_level", type=str, default="batch_seq",
                        choices=['batch', 'batch_seq', 'batch_seq_feature'])  # impute和mtand
    parser.add_argument("--ts_update_layer", type=str, default="gru", choices=['gru', 'attn'])

    # mtand
    parser.add_argument("--ts_mtand_value_transfer", type=bool, default=False)  # TODO：mtand中的值是否转移
    parser.add_argument("--mtand_steps", type=int, default=1)

    # 为了模型增加的
    parser.add_argument("--dropout", type=float, default=0.1)

    return parser.parse_args()