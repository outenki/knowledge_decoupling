# simplify_preserve_tense.py
import spacy
import nltk
import argparse
from pathlib import Path
from functools import partial
import os
import ssl
from functools import lru_cache

from datasets.dataset_dict import DatasetDict
from datasets.arrow_dataset import Dataset
from nltk.corpus import wordnet as wn
from lemminflect import getInflection

from lib.basic_words.basic_words_850 import BASIC_WORDS_850
from lib.basic_words.oxford_3000 import OXFORD_3000
from lib.dataset import load_custom_dataset, slice_dataset, skip_dataset_by_column
from lib.text import simple_split_text


try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Python < 2.7.9 没有 ssl._create_unverified_context
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('wordnet')
nltk.download('omw-1.4')

BASIC_VOCAB = OXFORD_3000
NUM_PROC = min(4, os.cpu_count())

_nlp = None

def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def spacy_to_wordnet_pos(spacy_pos):
    if spacy_pos.startswith("N"):
        return wn.NOUN
    elif spacy_pos.startswith("V"):
        return wn.VERB
    elif spacy_pos.startswith("J"):
        return wn.ADJ
    elif spacy_pos.startswith("ADJ"):
        return wn.ADJ
    elif spacy_pos.startswith("R"):
        return wn.ADV
    elif spacy_pos.startswith("ADV"):
        return wn.ADV
    else:
        return None


@lru_cache(maxsize=100_000)
def get_synsets_cached(lemma, pos):
    return wn.synsets(lemma, pos=pos)


def get_simple_candidate(token):
    if token.is_punct or token.is_space:
        return token.text

    # skip numbers, PROPN and PRON
    if token.pos_ == "PROPN" or token.pos_ == "PRON" or token.like_num:
        return token.text

    # skip NEs
    if token.ent_type_:
        return token.text

    lemma = token.lemma_.lower()
    if lemma in BASIC_VOCAB:
        return lemma

    candidates = []
    spacy_pos = spacy_to_wordnet_pos(token.pos_)

    for syn in get_synsets_cached(lemma, pos=spacy_pos):
        if not syn:
            continue
        for sn in syn.lemma_names():
            if "_" in sn:
                continue
            if sn in BASIC_VOCAB:
                candidates.append(sn)
    candidates = list(set(candidates))

    if not candidates:
        return None

    best_cand = None
    best_sim = -1.0
    for cand in candidates:
        for cand_syn in get_synsets_cached(cand, pos=spacy_pos):
            if not cand_syn:
                continue
            for word_syn in get_synsets_cached(lemma, pos=spacy_pos):
                if not word_syn:
                    continue
                try:
                    sim = cand_syn.wup_similarity(word_syn)
                except Exception:
                    sim = None
                if sim and sim > best_sim:
                    best_sim = sim
                    best_cand = cand

    # take the shortest one
    if best_cand:
        # best_cand = sorted(candidates, key=lambda x: (len(x.split()), len(x)))[0]
        best_cand = candidates[0]
    return best_cand


def inflect_candidate(lemma, target_tag):
    if lemma is None:
        return None
    valid_tags = set([
        "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "NN", "NNS",
        "NNP", "NNPS", "JJ", "JJR", "JJS", "RB", "RBR", "RBS"
    ])
    if target_tag not in valid_tags:
        return lemma
    infl = getInflection(lemma, tag=target_tag)
    if infl:
        return infl[0]
    return lemma


def safe_texts(texts, max_len):
    for t in texts:
        if len(t) <= max_len:
            yield t
        else:
            yield ""


def simplify_long_text(text: str):
    # split text to sents
    sents = simple_split_text(text)
    nlp = get_nlp()
    docs = nlp.pipe(safe_texts(sents, nlp.max_length), batch_size=128, n_process=4)
    simplified = [simplify_sentence(doc) for doc in docs]
    return ' '.join(simplified)

def simplify_sentences(examples):
    text = [simplify_long_text(example) for example in examples]
    return {
        "ori_text": examples,
        "text": text
    }


def simplify_sentence(doc):
    out_tokens = []
    for token in doc:
        if str(token).lower() in ("be", "am", "are", "is", "were", "was", "been"):
            out_tokens.append(token.text_with_ws)
            continue

        candidate = get_simple_candidate(token)
        if candidate is None:
            return ""
            # out_tokens.append(token.text_with_ws)
            # continue

        inflected = inflect_candidate(candidate, token.tag_)
        if inflected is None:
            inflected = candidate

        if token.text[0].isupper():
            inflected = inflected.capitalize()

        out_tokens.append(inflected + token.whitespace_)
    return "".join(out_tokens)


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-name', '-dn', dest='data_name', type=str,
        help='Dataset path to load from.'
    )
    parser.add_argument(
        '--data-type', '-dt', dest='data_type', type=str, required=False, default=None,
        help=(
            'Type of the dataset to load. '
            'If not provided, the dataset will be loaded as a Hugging Face Dataset.'
        )
    )
    parser.add_argument(
        '--load-from', '-lf', dest='load_from', choices=["hf", "local"],
        help='Load dataset from Hugging Face or local path.'
    )
    parser.add_argument(
        '--start-from', '-sf', dest='start_from', type=int, default=0, required=False,
        help='Load data from line.'
    )
    parser.add_argument(
        '--limit', '-l', dest='data_limit', type=int, default=0, required=False,
        help='Limit the number of samples to process. 0 means no limit.'
    )
    parser.add_argument(
        '--basic-vocab', '-bv', dest='basic_vocab', type=str, choices={"bw850", "ox3000"},
    )
    parser.add_argument(
        '--skip-key', '-sk', dest='skip_key',
        help='Skip data based on column'
    )
    parser.add_argument(
        '--skip-values', '-sv', nargs='+', dest='skip_values',
        help='Skip data based on values of the skip_key'
    )
    parser.add_argument(
        '--out-path', '-o', dest='out_path', type=str,
        help='Path to save the dataset with nonce sentences.'
    )
    return parser.parse_args()


def main():
    args = read_args()
    print(vars(args))
    out_path = args.out_path
    Path(out_path).mkdir(parents=True, exist_ok=True)

    # ========  Load dataset ========
    print("**** Loading dataset...")
    dataset = load_custom_dataset(
        data_name=args.data_name,
        data_type=args.data_type,
        load_from=args.load_from
    )
    print(f"Dataset loaded with {dataset.num_rows} samples.")
    global BASIC_VOCAB
    if args.basic_vocab == "bw850":
        BASIC_VOCAB = BASIC_WORDS_850
    if args.basic_vocab == "ox3000":
        BASIC_VOCAB = OXFORD_3000

    # ======== simplify sentences ========
    if isinstance(dataset, DatasetDict):
        dataset_dict = {}
        for key, dt in dataset.items():
            dt_limit = args.data_limit if key == "train" else int(args.data_limit * 0.1)
            if args.skip_key and args.skip_values:
                dt = skip_dataset_by_column(dt, args.skip_key, args.skip_values)
            start_from = args.start_from if key == "train" else int(args.start_from * 0.1) 
            dt = slice_dataset(dt, start_from, dt_limit)
            print(f"========= Processing dataset {key}... ==========")
            print(f"Dataset {key} has {dt.num_rows} samples after slicing.")
            process_fn = partial(simplify_sentences)
            dataset_dict[key] = dt.map(
                process_fn,
                num_proc=NUM_PROC,
                batched=True,
                writer_batch_size=1000,
                input_columns=["text"],
                desc="Simplifying sentences"
            )
            dataset_dict[key] = dataset_dict[key].filter(lambda x: x["text"] is not None)
        if "train" in dataset_dict:
            dataset_dict["train"].select(range(5)).to_json(Path(out_path) / "example_nonce_sent.json")
        print(f"Saving dataset with nonce sentences to {out_path}...")
        dataset_dict = DatasetDict(dataset_dict)
        print("Dataset structure:", dataset_dict)
        dataset_dict.save_to_disk(out_path)
    elif isinstance(dataset, Dataset):
        print("**** Processing dataset ...")
        dataset = slice_dataset(dataset, args.start_from, args.data_limit)
        if args.skip_key and args.skip_values:
            dataset = skip_dataset_by_column(dataset, args.skip_key, args.skip_values)
        print(f"Dataset has {dataset.num_rows} samples after slicing.")
        process_fn = partial(simplify_sentences)
        dataset = dataset.map(
            process_fn,
            num_proc=NUM_PROC,
            batched=True,
            writer_batch_size=1000,
            input_columns=["text"],
            desc="Simplifying sentences"
        )
        dataset = dataset.filter(lambda x: x["text"] is not None)
        print(f"Dataset has {dataset.num_rows} samples after generating nonce sentences.")
        print(f"Saving dataset with nonce sentences to {out_path}...")
        dataset.save_to_disk(out_path)
    else:
        raise TypeError


if __name__ == "__main__":
    main()
