import pickle
import random

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image



class MIMIC3Ihm48Dataset(Dataset):
    def __init__(self, args, data_path, device, note_length, note_num=10, bert="bioLongformer", ts_max_len=200):
        self.args = args
        self.baseline_model = args.baseline_model
        self.data = self.read_pkl(data_path)
        self.note_length = str(note_length)
        self.note_num = note_num
        self.bert = bert
        self.K = args.var_dim
        self.ts_max_len = ts_max_len
        self.mtand_query_tt_steps = args.mtand_steps
        self.conti_impute_type = args.conti_impute_type
        if bert == 'bioLongformer':
            self.note_index = 'note_encode_' + self.note_length + '_data'
            self.empty = torch.tensor([0, 2])
            self.pad = 1

    def read_pkl(self, path):
        with open(path, 'rb') as file:
            data = pickle.load(file)
        data_modality = self.args.data_modality
        if data_modality == 'ts':
            new_data = []
            for item in data:
                if item['modality'] == 'TS' or item['modality'] == 'TS_Note':
                    new_data.append(item)
        elif data_modality == 'ns':
            new_data = []
            for item in data:
                if item['modality'] == 'NS' or item['modality'] == 'TS_Note':
                    new_data.append(item)
        else:
            new_data = data
        return new_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # index = 1742
        # print("index:", index)
        self.now_data = self.data[index]
        # print(self.now_data)
        name = self.now_data['name']

        age = self.now_data['age']
        gender = self.now_data['gender']
        demogra = torch.tensor([age, gender])

        modality = self.now_data['modality']
        if modality == "TS":
            ns_missing = torch.tensor(1).float()  # 1表示模态缺失
            ts_missing = torch.tensor(0).float()
        elif modality == "Note":
            ns_missing = torch.tensor(0).float()
            ts_missing = torch.tensor(1).float()
        else:
            ns_missing = torch.tensor(0).float()
            ts_missing = torch.tensor(0).float()

        # 0. 读取数据
        if modality != "Note":
            ts_data = torch.tensor(self.now_data['ts_data']).float()
            ts_mask = torch.tensor(self.now_data['ts_mask']).float()
            ts_tt = torch.tensor(self.now_data['ts_tt']).float()
            ts_tau = torch.tensor(self.now_data['ts_tau']).float()
        else:
            ts_data = torch.zeros(1, self.K).float()
            ts_mask = torch.zeros(1, self.K).float()
            ts_tt = torch.zeros(1).float()
            ts_tau = torch.zeros(1, self.K).float()

        # 2. 若时间序列长度大于ts_max_len，则截取一下最后的时刻
        if ts_data.shape[0] > self.ts_max_len:
            ts_data, ts_mask, ts_tt, ts_tau = self.cut_ts_data(ts_data, ts_mask, ts_tt, ts_tau)

        if modality != "TS":
            note_data = self.now_data['note_token']
            note_tt = self.now_data['note_tt']
            note_tau = self.now_data['note_tau']
            # 3. 选取最后5段note
            note_data, note_tt, note_tau, note_mask = self.choose_note(note_data, note_tt, note_tau)

        else:
            note_data = [self.empty for _ in range(self.note_num)]
            note_tt = [0] * self.note_num
            note_tau = [0] * self.note_num
            note_mask = [0] * self.note_num
            note_mask = torch.tensor(note_mask).float()

        label = torch.tensor(self.now_data['label']).float()

        # 4. 将note补0-->tensor
        note_data = pad_sequence(note_data, batch_first=True, padding_value=self.pad)
        # [5, maxlen] --> [maxlen, 5]
        note_data = note_data.transpose(1, 0)
        note_tt = torch.tensor(note_tt).float()
        note_tau = torch.tensor(note_tau).float()

        # 5. query tt
        query_tt = torch.arange(0, 48.0001, self.mtand_query_tt_steps).float()

        # 6. impute query ts data
        query_ts_data = self.impute_ts_data(ts_data, ts_mask, ts_tt)

        return name, demogra, ts_data, ts_tt, ts_mask, ts_tau, note_data, note_tt, note_mask, note_tau, label, query_tt, query_ts_data, ts_missing, ns_missing, self.pad

    def pad_to_max_len(self, ts_data, ts_mask, ts_tt, ts_tau):
        L, K = ts_data.shape
        if L >= self.ts_max_len:
            return ts_data, ts_mask, ts_tt, ts_tau

        pad_data = torch.zeros(self.ts_max_len - L, K).float()
        pad_time = torch.zeros(self.ts_max_len - L).float()
        ts_data = torch.cat([ts_data, pad_data], dim=0)
        ts_mask = torch.cat([ts_mask, pad_data], dim=0)
        ts_tau = torch.cat([ts_tau, pad_data], dim=0)
        ts_tt = torch.cat([ts_tt, pad_time], dim=0)
        return ts_data, ts_mask, ts_tt, ts_tau

    def cut_ts_data(self, ts_data, ts_mask, ts_tt, ts_tau):
        # 截取
        ts_data = ts_data[-self.ts_max_len:]
        ts_mask = ts_mask[-self.ts_max_len:]
        ts_tt = ts_tt[-self.ts_max_len:]
        ts_tau = ts_tau[-self.ts_max_len:]
        return ts_data, ts_mask, ts_tt, ts_tau

    def choose_note(self, note_data, note_tt, note_tau):
        # 选取5段最后；若不够5段，则需要补充
        length = self.note_num
        note_len = len(note_data)
        note_mask = torch.ones(length)
        if note_len < length:
            padding_num = length - note_len
            empty_note = [self.empty for _ in range(padding_num)]
            note_data = note_data + empty_note
            note_tt = np.concatenate((note_tt, [0]*padding_num))
            note_tau = np.concatenate((note_tau, [0]*padding_num))
            note_mask[-padding_num:] = 0
        elif note_len > length:
            note_data = note_data[-length:]
            note_tt = note_tt[-length:]
            note_tau = note_tau[-length:]

        return note_data, note_tt, note_tau, note_mask

    def impute_ts_data(self, ts_data, ts_mask, ts_tt):
        if self.args.task == "24pheno":
            tt_max = 24
            duration = self.mtand_query_tt_steps
        elif self.args.task == "48ihm":
            tt_max = 48
            duration = self.mtand_query_tt_steps

        no_feature = ts_data.shape[1]  # K
        impute = torch.zeros((int(tt_max // duration)+1, no_feature * 2))
        for x, t, m in zip(ts_data, ts_tt, ts_mask):  # 每个时刻
            row = int(t / duration)
            if row >= tt_max:
                continue
            for f_idx, (rwo_x, row_m) in enumerate(zip(x, m)):  # 每个维度
                if row_m == 1:
                    impute[row][no_feature + f_idx] = 1
                    impute[row][f_idx] = rwo_x
                else:
                    if impute[row - 1][f_idx] != 0:
                        impute[row][f_idx] = impute[row - 1][f_idx]

        return impute


def MIMIC3_ihm_collate_fn(batch):
    names, demogras, ts_datas, ts_tts, ts_masks, ts_taus, note_datas, note_tts, note_masks, note_taus, label, query_tts, query_ts_data, ts_miss, ns_miss, pads = zip(*batch)

    pad_list = list(pads)
    pad = pad_list[0]

    # 转换为列表
    names = list(names)  # list
    # note_datas = list(note_datas)
    # note_tts = list(note_tts)
    # note_masks = list(note_masks)
    # note_taus = list(note_taus)

    demogras = torch.stack(demogras)  # torch

    ts_datas = pad_sequence(ts_datas, batch_first=True, padding_value=0)  # torch
    ts_tts = pad_sequence(ts_tts, batch_first=True, padding_value=0)  # torch
    ts_masks = pad_sequence(ts_masks, batch_first=True, padding_value=0)  # torch
    ts_taus = pad_sequence(ts_taus, batch_first=True, padding_value=0)  # torch

    note_datas = list(note_datas)
    note_datas = pad_sequence(note_datas, batch_first=True, padding_value=pad)  # torch
    # print("note_datas.shape:", note_datas.shape)
    note_datas = note_datas.transpose(2, 1)
    # print("note_datas.shape:", note_datas.shape)

    note_attention_mask = torch.zeros_like(note_datas)
    note_attention_mask[note_datas != pad] = 1
    # print("note_datas_attention_mask.shape:", note_datas_attention_mask)

    note_token_type = torch.zeros_like(note_datas)

    note_tts = torch.stack(note_tts)
    note_taus = torch.stack(note_taus)
    note_masks = torch.stack(note_masks)
    label = torch.stack(label)

    query_tts = torch.stack(query_tts)
    query_ts_data = torch.stack(query_ts_data)

    ts_miss = torch.stack(ts_miss)
    ns_miss = torch.stack(ns_miss)

    return names, demogras, ts_datas, ts_tts, ts_masks, ts_taus,  note_datas, note_attention_mask, note_token_type, note_tts, note_taus, note_masks, query_tts, query_ts_data, ts_miss, ns_miss, label


from baseline_config.UTDE_MIMIC3_48ihm_args_config import UTDE_MIMIC3_48ihm_parse_args


if __name__ == '__main__':
    # import torch.multiprocessing as mp
    # mp.set_start_method('spawn', force=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seed_value = 68  # 可以替换为任何你想要的整数种子值
    random.seed(seed_value)

    # 创建CustomDataset实例
    args = UTDE_MIMIC3_48ihm_parse_args()

    args.baseline_model = "mTand_txt"
    dataset = MIMIC3Ihm48Dataset(args=args, data_path='../data/MIMIC3/merge/test_normed_full.pkl', device=device, note_length=512, note_num=10, bert="bioLongformer")

    print("len:", len(dataset))

    # 创建DataLoader实例，并使用自定义的collate_fn函数
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=MIMIC3_ihm_collate_fn, num_workers=2)

    sum = 0
    # 使用DataLoader迭代数据
    for batch in dataloader:
        name, demogra, ts_data, ts_tt, ts_mask, ts_tau, note_data, note_attention_mask, note_token_type, note_tt, note_tau, note_mask, query_tt, query_ts_data, ts_miss, ns_miss, label = batch

        print("--------------------START---------------------------")
        print("ts_data:", ts_data.shape)

        # print(f"note data shape:{note_data.shape}")
        # print("note_data:", note_data[:2, :2])

        print("ts_miss:", ts_miss)
        print("ns_miss:", ns_miss)

        print("ts_data_miss:", ts_data[ts_miss==1, :2])
        print("ts_tt:", ts_tt)
        print("note_data_miss:", note_data[ns_miss==1, :2])

        print("label:", label.shape)
        print("label:", label)

    print("sum:", sum)