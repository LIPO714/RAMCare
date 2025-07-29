import pickle

from tqdm import tqdm


def read_pkl(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def save_to_pkl(data_dict, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


def data_merge(state):
    full_data_path = f"../data/MIMIC3/merge/{state}_normed_full.pkl"
    ts_data_path = f"../data/MIMIC3/analysis/UTDE_MIMIC3_48ihm_{state}_analysis.pkl"
    txt_data_path = f"../data/MIMIC3/analysis/mTand_txt_MIMIC3_48ihm_{state}_analysis.pkl"
    save_path = f"../data/MIMIC3/analysis/merged_analysis_128_256.pkl"
    full_data = read_pkl(full_data_path)
    ts_data = read_pkl(ts_data_path)
    txt_data = read_pkl(txt_data_path)

    merge_data_dict = []

    for item in tqdm(full_data):
        name = item['name']
        item['ts_hs'] = ts_data[name]
        item['note_hs'] = txt_data[name]
        merge_data_dict.append(item)

    save_to_pkl(merge_data_dict, save_path)

    return merge_data_dict


if __name__ == '__main__':
    all_data = []
    all_data = data_merge('train')

