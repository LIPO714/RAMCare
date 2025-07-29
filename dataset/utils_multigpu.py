import pickle

from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from torch.utils.data.distributed import DistributedSampler


from dataset.MIMIC3ihmDataset import MIMIC3Ihm48Dataset, MIMIC3_ihm_collate_fn
from dataset.MIMIC4ihmDataset import MIMIC4IhmDataset, MIMIC4_ihm_collate_fn


def data_perpare(args, task, device, mode='train'):
    if task == "48ihm":
        if mode == 'train':
            data_path = args.ihm_train_data + args.data_percent + '.pkl'
            if args.baseline_model in ["mTand_txt", 'RAGCare', 'RAGCare_ablation', 'RAGCare_casestudy']:
                dataset = MIMIC3Ihm48Dataset(args=args, data_path=data_path, device=device, note_length=args.max_length, note_num=args.num_of_notes, bert=args.model_name, ts_max_len=args.ts_max_len)
                sampler = DistributedSampler(dataset)
                dataloader = DataLoader(dataset=dataset,
                                        batch_size=args.train_batch_size,
                                        shuffle=False,
                                        num_workers=8,
                                        pin_memory=False,
                                        sampler=sampler,
                                        collate_fn=MIMIC3_ihm_collate_fn)
            elif args.baseline_model in ['MIMIC4_TextModel', 'RAGCare_MIMIC4']:
                dataset = MIMIC4IhmDataset(args=args, data_path=data_path, device=device, note_length=args.max_length,
                                             note_num=args.num_of_notes, bert=args.model_name,
                                             ts_max_len=args.ts_max_len)
                sampler = DistributedSampler(dataset)
                dataloader = DataLoader(dataset=dataset,
                                        batch_size=args.train_batch_size,
                                        shuffle=False,
                                        num_workers=8,
                                        pin_memory=False,
                                        sampler=sampler,
                                        collate_fn=MIMIC4_ihm_collate_fn)
        elif mode == 'val':
            if args.baseline_model in ['RAGCare_casestudy']:
                data_path = args.ihm_val_data + 'full.pkl'
            else:
                if args.stage == "finetune":
                    data_path = args.ihm_val_data + 'all.pkl'
                else:
                    data_path = args.ihm_val_data + args.data_percent + '.pkl'
            if args.baseline_model in ["mTand_txt", 'RAGCare', 'RAGCare_ablation', 'RAGCare_casestudy']:
                dataset = MIMIC3Ihm48Dataset(args=args, data_path=data_path, device=device, note_length=args.max_length, note_num=args.num_of_notes, bert=args.model_name, ts_max_len=args.ts_max_len)
                sampler = DistributedSampler(dataset)
                dataloader = DataLoader(dataset=dataset,
                                        batch_size=args.eval_batch_size,
                                        shuffle=False,
                                        num_workers=8,
                                        pin_memory=False,
                                        sampler=sampler,
                                        collate_fn=MIMIC3_ihm_collate_fn)
            elif args.baseline_model in ['MIMIC4_TextModel', 'RAGCare_MIMIC4']:
                dataset = MIMIC4IhmDataset(args=args, data_path=data_path, device=device, note_length=args.max_length,
                                             note_num=args.num_of_notes, bert=args.model_name,
                                             ts_max_len=args.ts_max_len)
                sampler = DistributedSampler(dataset)
                dataloader = DataLoader(dataset=dataset,
                                        batch_size=args.eval_batch_size,
                                        shuffle=False,
                                        num_workers=8,
                                        pin_memory=False,
                                        sampler=sampler,
                                        collate_fn=MIMIC4_ihm_collate_fn)
        elif mode == 'test':
            if args.baseline_model in ['RAGCare_casestudy']:
                data_path = args.ihm_val_data + 'full.pkl'
            else:
                if args.stage == "finetune":
                    data_path = args.ihm_test_data + 'all.pkl'
                else:
                    data_path = args.ihm_test_data + args.data_percent + '.pkl'
            if args.baseline_model in ["mTand_txt", 'RAGCare', 'RAGCare_ablation', 'RAGCare_casestudy']:
                dataset = MIMIC3Ihm48Dataset(args=args, data_path=data_path, device=device, note_length=args.max_length,
                                          note_num=args.num_of_notes, bert=args.model_name, ts_max_len=args.ts_max_len)
                sampler = DistributedSampler(dataset)
                dataloader = DataLoader(dataset=dataset,
                                        batch_size=args.eval_batch_size,
                                        shuffle=False,
                                        num_workers=8,
                                        pin_memory=False,
                                        sampler=sampler,
                                        collate_fn=MIMIC3_ihm_collate_fn)
            elif args.baseline_model in ['MIMIC4_TextModel', 'RAGCare_MIMIC4']:
                dataset = MIMIC4IhmDataset(args=args, data_path=data_path, device=device, note_length=args.max_length,
                                             note_num=args.num_of_notes, bert=args.model_name,
                                             ts_max_len=args.ts_max_len)
                sampler = DistributedSampler(dataset)
                dataloader = DataLoader(dataset=dataset,
                                        batch_size=args.eval_batch_size,
                                        shuffle=False,
                                        num_workers=8,
                                        pin_memory=False,
                                        sampler=sampler,
                                        collate_fn=MIMIC4_ihm_collate_fn)

    return dataset, sampler, dataloader


def data_to_device(device, batch):
    name, demogra, ts_data, ts_tt, ts_mask, ts_tau, note_data, note_attention_mask, note_token_type, note_tt, note_tau, note_mask, query_tt, query_ts_data, ts_miss, ns_miss, label = batch
    demogra = demogra.to(device)
    ts_data = ts_data.to(device)
    ts_tt = ts_tt.to(device)
    ts_mask = ts_mask.to(device)
    ts_tau = ts_tau.to(device)
    note_data = note_data.to(device)
    note_attention_mask = note_attention_mask.to(device)
    note_token_type = note_token_type.to(device)
    note_tt = note_tt.to(device)
    note_tau = note_tau.to(device)
    note_mask = note_mask.to(device)
    query_tt = query_tt.to(device)
    query_ts_data = query_ts_data.to(device)
    ts_miss = ts_miss.to(device)
    ns_miss = ns_miss.to(device)
    label = label.to(device)
    return name, demogra, ts_data, ts_tt, ts_mask, ts_tau, note_data, note_attention_mask, note_token_type, note_tt, note_tau, note_mask, query_tt, query_ts_data, ts_miss, ns_miss, label


def read_rag(args, type):
    if type == "ts2ns":
        path = args.ts2ns_rag
        cluster_num = args.ts_cluster_num
        lambda_align = args.ts_lambda_align
    else:
        path = args.ns2ts_rag
        cluster_num = args.ns_cluster_num
        lambda_align = args.ns_lambda_align

    path = path + f"{cluster_num}_{lambda_align}.pkl"
    with open(path, 'rb') as file:
        data = pickle.load(file)
    if type == "ts2ns":
        K = data["ts_2_note_from"]
        V = data["ts_2_note_to"]
    else:  # ns2ts
        K = data["note_2_ts_from"]
        V = data["note_2_ts_to"]
    return K, V

def read_ablation_wo_cokeans_rag(args, type):
    if type == "ts2ns":
        path = args.ts2ns_rag
        cluster_num = args.ts_cluster_num
        lambda_align = 0.0
    else:
        path = args.ns2ts_rag
        cluster_num = args.ns_cluster_num
        lambda_align = 0.0

    path = path + f"{cluster_num}_{lambda_align}.pkl"
    with open(path, 'rb') as file:
        data = pickle.load(file)
    if type == "ts2ns":
        K = data["ts_2_note_from"]
        V = data["ts_2_note_to"]
    else:  # ns2ts
        K = data["note_2_ts_from"]
        V = data["note_2_ts_to"]
    return K, V
