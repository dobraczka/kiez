"""
Convenience methods for loading entity embeddings from knowledge graph embeddings
"""
import os
from typing import Dict, Tuple

import numpy as np


def _read_kg_ids(path) -> Dict[int, str]:
    with open(path) as in_file:
        return {
            int(line.strip().split("\t")[1]): line.strip().split("\t")[0]
            for line in in_file
        }


def _read_ent_links(path) -> Dict[str, str]:
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


def _read_openea_files(emb_dir_path, kg_path):
    emb = np.load(os.path.join(emb_dir_path, "ent_embeds.npy"))
    kg1_ids = _read_kg_ids(os.path.join(emb_dir_path, "kg1_ent_ids"))
    kg2_ids = _read_kg_ids(os.path.join(emb_dir_path, "kg2_ent_ids"))
    ent_links = _read_ent_links(os.path.join(kg_path, "ent_links"))
    return emb, kg1_ids, kg2_ids, ent_links


def _seperate_common_embedding(
    emb: np.ndarray,
    kg1_ids: Dict[int, str],
    kg2_ids: Dict[int, str],
    ent_links: Dict[str, str],
) -> Tuple[np.ndarray, np.ndarray, Dict[int, str], Dict[int, str], Dict[str, str]]:
    """Seperate single embedding array into two arrays split by knowledge graph
    Parameters
    ----------
    emb: np.ndarray
        embedding array
    kg1_ids: Dict[int,str]
        row index and entity ids of knowledge graph 1
    kg2_ids: Dict[int,str]
        row index and entity ids of knowledge graph 2
    ent_links: Dict[str,str]
        linked entities

    Returns
    -------
    emb1, emb2, kg1_ids_new, kg2_ids_new, ent_links_new
        embeddings of both knowledge graphs
        entity ids per embedding row for both knowledge graphs
        entity links
    """
    emb1, kg1_ids_new = _split_emb(emb, kg1_ids)
    emb2, kg2_ids_new = _split_emb(emb, kg2_ids)
    ent_links_new = {kg1_ids_new[e1]: kg2_ids_new[e2] for e1, e2 in ent_links.items()}
    return emb1, emb2, kg1_ids_new, kg2_ids_new, ent_links_new


def from_openea(
    emb_dir_path: str, kg_path: str
) -> Tuple[np.ndarray, np.ndarray, Dict[int, str], Dict[int, str], Dict[int, int]]:
    """Load OpenEA-type data
    Parameters
    ----------
    emb_dir_path: str
        Path to folder containing embedding
    kg_path: str
        Path to folder containing kg info

    Returns
    -------
    emb1, emb2, kg1_ids_new, kg2_ids_new, ent_links_new
        embeddings of both knowledge graphs
        entity ids per embedding row for both knowledge graphs
        entity links
    Notes
    -----
    See here for more information on the dataset structure
    https://github.com/nju-websoft/OpenEA#dataset-description
    """
    return _seperate_common_embedding(*_read_openea_files(emb_dir_path, kg_path))
