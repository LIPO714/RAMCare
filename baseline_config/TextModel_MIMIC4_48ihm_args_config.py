import argparse



def TextModel_MIMIC4_48ihm_parse_args():
    parser = argparse.ArgumentParser(description="argparse")

    parser.add_argument("--exp_name", type=str, default="TextModel_MIMIC4_48ihm_dim256_all_analysis_1009")
    parser.add_argument("--stage", type=str, default="single_modal")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--task", type=str, default='48ihm')  # 48ihm 24pheno
    parser.add_argument("--mode", type=str, default='train', choices=["train", "eval"])  # train eval
    parser.add_argument("--fp16", type=bool, default=True)
    parser.add_argument("--tensorboard_dir", type=str, default='./tensorboard/')
    parser.add_argument("--save_dir", type=str, default='./save/')
    parser.add_argument("--log_dir", type=str, default='./log/')

    parser.add_argument('--init_method', default='tcp://localhost:18888', help="init-method")
    parser.add_argument('-g', '--gpuid', default=1, type=int, help="which gpu to use")  # 第几号卡
    parser.add_argument('-r', '--rank', default=0, type=int, help='rank of current process')  # 第几个进程
    parser.add_argument('-w', '--world_size', default=1, type=int, help="world size")  # 一共几张卡
    parser.add_argument('--use_mix_precision', default=False, action='store_true', help="whether to use mix precision")

    # data
    parser.add_argument("--var_dim", type=int, default=17)
    parser.add_argument("--invar_dim", type=int, default=2)
    parser.add_argument("--cate_dim", type=int, default=0)  # TODO: cate new
    parser.add_argument("--cate_type", type=str, default='4_6_13_5')  # TODO: cate new
    parser.add_argument("--metadata", type=str, default='./data/metadata.json')

    parser.add_argument("--ihm_train_data", type=str, default='./data/MIMIC4/merge/train_')
    parser.add_argument("--ihm_val_data", type=str, default='./data/MIMIC4/merge/val_')
    parser.add_argument("--ihm_test_data", type=str, default='./data/MIMIC4/merge/test_')

    parser.add_argument("--data_modality", type=str, default="ns", choices=['ts', 'ns', 'all'])

    # ts data
    parser.add_argument("--ts_max_len", type=int, default=10)  # 220
    parser.add_argument("--impute", default=True)
    parser.add_argument("--conti_impute_type", type=str, default="backward", choices=["linear", "backward",
                                                                                    "forward"])  # TODO：插值方法。backward 后面的时刻用前面的有效值插值；forward 前面的时刻用后面的有效值插值
    parser.add_argument("--cate_impute_type", type=str, default="backward", choices=["linear", "backward",
                                                                                     "forward"])  # TODO：插值方法。backward 后面的时刻用前面的有效值插值；forward 前面的时刻用后面的有效值插值

    # note data
    parser.add_argument("--notes_order", type=str, default='Last')
    parser.add_argument("--num_of_notes", type=int, default=1)  # 10
    parser.add_argument("--model_name", type=str, default="bioLongformer")
    parser.add_argument("--max_length", type=int, default=512)  # bioLongformer: 512 or 1024; BioBert: 256 or 512

    # optim
    parser.add_argument("--bert_opt", type=str, default="AdamW")
    parser.add_argument("--other_opt", type=str, default="AdamW")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--bert_lr", type=float, default=0.00004)
    parser.add_argument("--other_lr", type=float, default=0.0004)

    parser.add_argument("--variance_loss", type=bool, default=False)  # TODO: 是否包含variance_maximization_loss，用于分散化hidden state表征
    parser.add_argument("--variance_loss_lambda", type=float, default=0.5)

    # sche
    parser.add_argument("--bert_sche", type=str, default="linear_with_warmup")
    parser.add_argument("--other_sche", type=str, default="None")
    parser.add_argument("--warm_up", type=float, default=0.08)  # 0.08 * total_steps

    # train
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--modeltype", type=str, default='TS_Text')
    parser.add_argument("--before_update_bert_epochs", type=int, default=0)  # TODO：bert更新策略 前几个epoch不更新bert
    parser.add_argument("--num_update_bert_epochs", type=int, default=2)  # 间隔几个epoch更新一次bert
    parser.add_argument("--bertcount", type=int, default=3)  # 最多更新几次bert
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)  # 梯度累加 累计多少个step之后更新一次参数

    # eval
    parser.add_argument("--eval", type=bool, default=False)  # TODO: False:不包含eval
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=15)

    # test & analysis
    parser.add_argument("--full_model", type=str, default="./save/48ihm/TextModel_MIMIC4_48ihm_dim256_all_0626_seed_42_epoch_50_batch_16_bioLongformer512_emb_dim_256/4.pth")  # "./save/48ihm/mTand_txt_MIMIC3_48ihm_0430_seed_42_epoch_50_batch_3_bioLongformer512_emb_dim_128/2.pth"
    parser.add_argument("--analysis_data", type=str, default="./data/MIMIC4/analysis/")

    # model
    parser.add_argument("--ns_embed_dim", type=int, default=256)
    parser.add_argument("--remove_rep", type=str, default="None")  # "None" 'time' 'type' 'density' 'invar'
    parser.add_argument("--mixup_level", type=str, default="batch_seq",
                        choices=['batch', 'batch_seq', 'batch_seq_feature'])  # impute和mtand
    parser.add_argument("--mtand_steps", type=int, default=1)

    # mtand
    parser.add_argument("--ns_mtand_value_transfer", type=bool, default=False)  # TODO：mtand中的值是否转移

    # model restore
    parser.add_argument("--ts_restore_len", type=int, default=3)  # l

    # model contrastive learning
    parser.add_argument("--contrastive_num", type=int, default=3)  # L_t

    # 为了模型增加的
    parser.add_argument("--dropout", type=float, default=0.1)

    # Loss = CONT_loss + lambda_1 * ts_restore_mse + lambda_2 * note_restore_cos
    parser.add_argument(
        "--loss",
        default="CONT+RESTORE",
        choices=["CONT", "RESTORE", "CONT+RESTORE"],
    )
    parser.add_argument("--temp", type=float, default=0.3)  # 用于初始化温度超参数，这个可以尝试调？
    parser.add_argument("--learnable_temp", action="store_true")
    parser.add_argument("--contrastive_emb_dim", type=float, default=0.5)  # if loss=="CONT+RESTORE"
    parser.add_argument("--lambda_1", type=float, default=1.0)  # TODO: 调节ts_restore的loss占比
    parser.add_argument("--lambda_2", type=float, default=1.0)  # TODO: 调节note_restore的loss占比

    return parser.parse_args()