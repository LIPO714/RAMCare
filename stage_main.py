import torch.nn as nn

from tensorboardX import SummaryWriter

import warnings
import logging

from analysis.analysis_amp import analysiser_amp_downstream
from dataset.utils_multigpu import data_perpare, read_ram, read_ablation_wo_cokeans_ram

from optim.optim_factory import create_optimizer, create_optimizer_2
from scheduler.scheduler_factory import create_scheduler, create_scheduler_2
from test import tester_downstream
from train_stage import trainer_downstream

logger = logging.getLogger(__name__)
# from model import *
# from train import *
from checkpoint import *
from util import *
# from interp import *

import torch.distributed as dist
from torch.cuda.amp import GradScaler

from RAMCare.RAMCare import RAMCare
from RAMCare_MIMIC4.RAMCare import RAMCareMIMIC4

from baseline_config.MIMIC3_48ihm_pretrain_args_config import MIMIC3_48ihm_pretrain_parse_args
from baseline_config.MIMIC3_48ihm_finetune_args_config import MIMIC3_48ihm_finetune_parse_args
from baseline_config.MIMIC4_48ihm_pretrain_args_config import MIMIC4_ihm_pretrain_parse_args
from baseline_config.MIMIC4_48ihm_finetune_args_config import MIMIC4_ihm_finetune_parse_args
from baseline_config.MIMIC3_ablation_48ihm_finetune_args_config import MIMIC3_ablation_finetune_parse_args
from baseline_config.MIMIC3_ablation_48ihm_pretrain_args_config import MIMIC3_ablation_pretrain_parse_args


def main():
    baseline_model = 'RAMCare'
    task = '48ihm'
    mode = 'train'
    stage = "pretrain"

    if baseline_model == 'RAMCare':
        if stage == "pretrain":
            args = MIMIC3_48ihm_pretrain_parse_args()
            args.data_percent = "full"
        elif stage == "finetune":
            args = MIMIC3_48ihm_finetune_parse_args()
            # args.data_percent = "all"
    elif baseline_model == 'RAMCare_MIMIC4':
        if stage == "pretrain":
            args = MIMIC4_ihm_pretrain_parse_args()
            args.data_percent = "full"
        elif stage == "finetune":
            args = MIMIC4_ihm_finetune_parse_args()
    elif baseline_model == 'RAMCare_ablation':
        if stage == "pretrain":
            args = MIMIC3_ablation_pretrain_parse_args()
            args.data_percent = "full"
        elif stage == "finetune":
            args = MIMIC3_ablation_finetune_parse_args()

    args.baseline_model = baseline_model
    args.mode = mode

    ########################################    N1    ####################################################################
    dist.init_process_group(backend='nccl', init_method=args.init_method, rank=args.rank, world_size=args.world_size)  #
    print("init process group")
    ######################################################################################################################

    print(args)
    make_save_dir(args)
    make_log_dir(args)
    make_tensorboard_dir(args)

    if torch.cuda.is_available():
        device = 'cuda:{}'.format(args.gpuid)
        gpu = args.gpuid
        # pass
    else:
        device = 'cpu'
    # device = 'cpu'

    args.device = device

    print("Device:", device)
    if args.tensorboard_dir!=None:
        writer = SummaryWriter(args.tensorboard_dir)
    else:
        writer=None

    warnings.filterwarnings('ignore')
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # ----------------Model---------------------

    classifier = None
    if baseline_model == 'RAMCare':
        BioBert, BioBertConfig, tokenizer = loadBert(args, device)
        model = RAMCare(args, BioBert)
    elif baseline_model == 'RAMCare_MIMIC4':
        BioBert, BioBertConfig, tokenizer = loadBert(args, device)
        model = RAMCareMIMIC4(args, BioBert)

    # ----------------Multi-GPU---------------------
    if device == 'cpu':
        model.cpu()
        if classifier is not None:
            classifier.cpu()
    else:
        model.cuda(gpu)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[gpu], find_unused_parameters=True
        )
        if classifier is not None:
            classifier.cuda(gpu)
            classifier = nn.SyncBatchNorm.convert_sync_batchnorm(classifier)
            classifier = torch.nn.parallel.DistributedDataParallel(
                classifier, device_ids=[gpu], find_unused_parameters=True
            )

    if args.mode == "train":
        if baseline_model == "RAMCare" and args.stage == "pretrain":
            print("--------------STAGE: Pretrain-----------------")
            ts_checkpoint = load_model(args.ts_pretrain_model, device)
            ns_checkpoint = load_model(args.ns_pretrain_model, device)
            model.module.TS_MODEL.load_state_dict(ts_checkpoint['model_state_dict'])

            raw_state_dict = ns_checkpoint['model_state_dict']
            clean_state_dict = {k.replace("module.", ""): v for k, v in raw_state_dict.items()}
            model.module.NS_MODEL.load_state_dict(clean_state_dict)

            # 冻结两个模态参数
            for param in model.module.TS_MODEL.parameters():
                param.requires_grad = False
            for param in model.module.NS_MODEL.parameters():
                param.requires_grad = False

            # 载入RAM, 并冻结RAM？
            ts2ns_K, ts2ns_V = read_ram(args, "ts2ns")
            ns2ts_K, ns2ts_V = read_ram(args, "ns2ts")
            model.module.TS2NS_Module.ram_module.init_KV(ts2ns_K, ts2ns_V)
            model.module.NS2TS_Module.ram_module.init_KV(ns2ts_K, ns2ts_V)

            for param in model.module.TS2NS_Module.ram_module.parameters():
                param.requires_grad = False
            for param in model.module.NS2TS_Module.ram_module.parameters():
                param.requires_grad = False

        elif baseline_model == "RAMCare_MIMIC4" and args.stage == "pretrain":
            print("--------------STAGE: Pretrain-----------------")
            ts_checkpoint = load_model(args.ts_pretrain_model, device)
            ns_checkpoint = load_model(args.ns_pretrain_model, device)
            model.module.TS_MODEL.load_state_dict(ts_checkpoint['model_state_dict'])

            raw_state_dict = ns_checkpoint['model_state_dict']
            clean_state_dict = {k.replace("module.", ""): v for k, v in raw_state_dict.items()}
            model.module.NS_MODEL.load_state_dict(clean_state_dict)

            # 冻结两个模态参数
            for param in model.module.TS_MODEL.parameters():
                param.requires_grad = False
            for param in model.module.NS_MODEL.parameters():
                param.requires_grad = False

            # 载入RAM, 并冻结RAM？
            ts2ns_K, ts2ns_V = read_ram(args, "ts2ns")
            model.module.TS2NS_Module.ram_module.init_KV(ts2ns_K, ts2ns_V)

            if args.ts_ram_freeze:
                for param in model.module.TS2NS_Module.ram_module.parameters():
                    param.requires_grad = False

        elif args.stage == "finetune":
            print("--------------STAGE: Finetune-----------------")
            checkpoint = load_model(args.pretrain_model, device)
            model.load_state_dict(checkpoint['model_state_dict'])

    else:
        # 读入训练参数
        checkpoint = load_full_model(args, device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if classifier is not None:
            classifier.load_state_dict(checkpoint['classifier'])

    # ----------------loss function-------------------
    if args.task == "48ihm":
        pred_loss_func = nn.BCEWithLogitsLoss(reduction='none').to(device)
        # pred_loss_func = nn.CrossEntropyLoss(reduction='none').to(device)

    if args.mode == "train":
        train_dataset, train_sampler, train_dataloader = data_perpare(args, args.task, device, mode='train')
        val_dataset, val_sampler, val_dataloader = data_perpare(args, args.task, device, mode='val')
        test_dataset, test_sampler, test_dataloader = data_perpare(args, args.task, device, mode='test')

        print(f"Train set size: {len(train_dataset)}")
        print(f"Validation set size: {len(val_dataset)}")
        print(f"Test set size: {len(test_dataset)}")

        args.steps_per_epoch = len(train_dataloader)

        class_optimizer = None
        class_scheduler = None
        if baseline_model in ["RAMCare", 'RAMCare_MIMIC4']:
            bert_optimizer, modal_optimizer, ram_optimizer, other_optimizer = create_optimizer_2(args, model)
            bert_scheduler, modal_scheduler, ram_scheduler, other_scheduler = create_scheduler_2(args, bert_optimizer, modal_optimizer, ram_optimizer, other_optimizer)


        trainer_downstream(model=model,args=args,train_dataloader=train_dataloader,val_dataloader=val_dataloader, test_dataloader=test_dataloader, device=device, loss_func=pred_loss_func, \
                bert_optimizer=bert_optimizer, modal_optimizer=modal_optimizer, ram_optimizer=ram_optimizer, other_optimizer=other_optimizer,  bert_scheduler=bert_scheduler, \
                             modal_scheduler=modal_scheduler, ram_scheduler=ram_scheduler, other_scheduler=other_scheduler, writer=writer, classifier=classifier, class_optimizer=class_optimizer, class_scheduler=class_scheduler)

        ############    N8    ###########
        dist.destroy_process_group()  #
        #################################


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))