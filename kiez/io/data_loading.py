import os
from typing import Dict

import numpy as np


def read_kg_ids(path) -> Dict[int, str]:
    with open(path) as in_file:
        return {
            int(line.strip().split("\t")[1]): line.strip().split("\t")[0]
            for line in in_file
        }


def read_ent_links(path) -> Dict[str, str]:
    with open(path) as in_file:
        return {
            line.strip().split("\t")[0]: line.strip().split("\t")[1] for line in in_file
        }


def _split_emb(emb, kg_ids):
    new_emb = []
    new_ids = {}
    i = 0
    for idx, e in enumerate(emb):
        if idx in kg_ids:
            new_ids[kg_ids[idx]] = i
            new_emb.append(e)
            i += 1
    return np.array(new_emb), new_ids


def read_openea_files(emb_dir_path, kg_path):
    emb = np.load(os.path.join(emb_dir_path, "ent_embeds.npy"))
    kg1_ids = read_kg_ids(os.path.join(emb_dir_path, "kg1_ent_ids"))
    kg2_ids = read_kg_ids(os.path.join(emb_dir_path, "kg2_ent_ids"))
    ent_links = read_ent_links(os.path.join(kg_path, "ent_links"))
    return emb, kg1_ids, kg2_ids, ent_links


def seperate_common_embedding(emb, kg1_ids, kg2_ids, ent_links):
    emb1, kg1_ids_new = _split_emb(emb, kg1_ids)
    emb2, kg2_ids_new = _split_emb(emb, kg2_ids)
    ent_links_new = {kg1_ids_new[e1]: kg2_ids_new[e2] for e1, e2 in ent_links.items()}
    return emb1, emb2, kg1_ids_new, kg2_ids_new, ent_links_new


def from_openea(emb_dir_path, kg_path):
    return seperate_common_embedding(*read_openea_files(emb_dir_path, kg_path))
