import argparse
import json
import re
from tqdm import tqdm
from pathlib import Path
from config import GPT_API_KEY
from openai import OpenAI
import random

from datasets import load_dataset, Dataset

client = OpenAI(api_key=GPT_API_KEY)


def print_args(args: dict):
    print("↓↓↓↓↓↓↓↓↓↓ Arguments ↓↓↓↓↓↓↓↓↓↓")
    for k, v in args.items():
        print(f"{k}: {v}")
    print("↑↑↑↑↑↑↑↑↑↑ Arguments ↑↑↑↑↑↑↑↑↑↑")


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-name', '-dn', dest='data_name', type=str, required=True,
        choices=['ai2_arc', 'boolq', 'qasc', "squad_v2", "mintaka", "cwq", "metaqa", "google_re", "commonsense_qa", "clasheval", "nq_swap"],
        help='Name of the dataset to load from Hugging Face'
    )
    parser.add_argument('--local-path', '-lp', dest='local_path', type=str)
    parser.add_argument('--split-name', '-sp', dest='split_name', type=str)
    parser.add_argument(
        '--subset-name', '-sn', dest='subset_name', type=str, required=False,
        default=None,
        help='Name of the subset'
    )
    parser.add_argument('--format', '-f', dest='format', action='store_true')
    parser.add_argument('--format-with-options', '-op', dest='format_with_options', action='store_true')
    parser.add_argument('--probing', '-p', dest='probing', action='store_true')
    parser.add_argument(
        '--context-conflict', '-cc', dest='context_conflict', type=str, choices=['ori', 'mod'], default='none',
        help='For context conflict dataset.'
    )
    parser.add_argument(
        '--context-key', '-ck', dest='context_key', type=str, required=False, default=None,
        help='Should be one of "snippet" or "considered_sentences". Only used when data_name is google_re.'
    )
    parser.add_argument(
        '--output-path', '-o', dest='output_path', type=str, required=True,
        help='Path to save results.'
    )
    return parser.parse_args()


def gpt_chat(prompt: str) -> str:
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        timeout=10
    )
    return response.output_text.strip()


def convert_qa_to_probing(question: str, answer: str) -> dict:
    gpt_prompt = "把问答句转换为填空题，确保答案出现在句末。 \n"
    gpt_prompt += "input: \nWhat is the last action that the student should perform before leaving the lab area? wash hands\n"
    gpt_prompt += "output: \n"
    gpt_prompt += "<text>: The last action that the student should perform before leaving the lab area is washing hands\n"
    gpt_prompt += "<question>: The last action that the student should perform before leaving the lab area is\n"
    gpt_prompt += "<answer>: washing hands\n\n"
    gpt_prompt += "input:\n" + f"{question} {answer}\n\n"
    resp = gpt_chat(gpt_prompt).strip()
    resp = re.sub(r'\n+', '\n', resp)
    text = resp.split("<text>:")[1].split("<question>:")[0].strip()
    question = resp.split("<question>:")[1].split("<answer>:")[0].strip()
    answer = resp.split("<answer>:")[1].strip()
    return {
        "text": text,
        "question": question,
        "answer": answer
    }


def construct_data(
    qid: str, context: str, question: str, options: list[str], answer: str,
    format: bool, format_with_options: bool, probing: bool,
    argkv: dict = {}
) -> dict:
    ori_context = context
    ori_question = question
    ori_options = options
    ori_answer = answer

    if probing:
        probing_text = convert_qa_to_probing(question, answer)
        text = probing_text["text"]
        question = probing_text["question"]
        answer = probing_text["answer"]
    else:
        text = question + " " + answer
    text = context + text

    if format:
        prompt = ""
        if context:
            prompt = f"### Context\n{context}"

        prompt += f"\n\n### Query\n{question}"
        if options and format_with_options:
            prompt += "\n\n### Options\n" + "\n".join(options)
        prompt += "\n\n Response\n"
    else:
        prompt = context + question

    result = {
        "id": qid,
        "ori_context": ori_context,
        "ori_question": ori_question,
        "ori_options": ori_options,
        "ori_answer": ori_answer,
        "prompt": prompt,
        "answer": answer,
        "text": text
    }

    if argkv:
        result.update(argkv)
    return result


def generate_qa_data_from_ai2_arc(
    dataset: Dataset, format: bool, format_with_options: bool, probing: bool
) -> list[dict]:
    qa_data = []
    for qid, sample in tqdm(enumerate(dataset), total=len(dataset), desc="Generating QA data"):
        assert isinstance(sample, dict)
        question = sample["question"]
        if not question.endswith("?") and not question.endswith("."):
            question += "?"

        labels = sample["choices"]["label"]
        answer_key = sample["answerKey"]
        options = sample["choices"]["text"]
        answer = options[labels.index(answer_key)]
        context = ""

        qa_data.append(
            construct_data(str(qid), context, question, options, answer, format, format_with_options, probing)
        )
    return qa_data


def generate_qa_data_from_boolq(
    dataset: Dataset, format: bool, format_with_options: bool, probing: bool
) -> list[dict]:
    qa_data = []
    for qid, sample in tqdm(enumerate(dataset), total=len(dataset), desc="Generating QA data"):
        assert isinstance(sample, dict)
        question = sample["question"]
        if not question.endswith("?") and not question.endswith("."):
            question += "?"
        answer = "Yes." if sample["answer"] else "No."
        context = sample["passage"]
        options = ["Yes", "No"]

        qa_data.append(
            construct_data(qid, context, question, options, answer, format, format_with_options, probing)
        )
    return qa_data


def generate_qa_data_from_qasc(
    dataset: Dataset, format: bool, format_with_options: bool, probing: bool
) -> list[dict]:
    qa_data = []
    for qid, sample in tqdm(enumerate(dataset), total=len(dataset), desc="Generating QA data"):
        assert isinstance(sample, dict)
        question = sample["question"]
        if not question.endswith("?") and not question.endswith("."):
            question += "?"
        options = sample["choices"]["text"]
        labels = sample["choices"]["label"]
        answer_key = sample["answerKey"]
        if answer_key not in labels:
            continue  # Skip if the answer key is not in the labels
        answer = options[labels.index(answer_key)]
        context = ""

        qa_data.append(
            construct_data(str(qid), context, question, options, answer, format, format_with_options, probing)
        )
    return qa_data


def generate_qa_data_from_squad(
    dataset: Dataset, format: bool, format_with_options: bool, probing: bool
) -> list[dict]:
    qa_data = []
    for qid, sample in tqdm(enumerate(dataset), total=len(dataset), desc="Generating QA data"):
        assert isinstance(sample, dict)
        question = sample["question"]
        context = sample["context"]
        answers = list(set(sample["answers"]["text"]))
        answer = answers[0] if answers else "I don't know."
        options = []

        qa_data.append(
            construct_data(str(qid), context, question, options, answer, format, format_with_options, probing)
        )
    return qa_data


def generate_qa_data_from_mintaka(
    dataset: list, format: bool, format_with_options: bool, probing: bool
) -> list[dict]:
    qa_data = []
    for sample in tqdm(dataset, total=len(dataset), desc="Generating QA data"):
        assert isinstance(sample, dict)
        qid: str = sample["id"]
        question = sample["question"]
        options = []
        answers = []
        if sample["complexityType"] != "multihop":
            continue
        if not sample["answer"]["answer"]:
            continue
        for _ans in sample["answer"]["answer"]:
            if not isinstance(_ans, dict):
                answers.append(str(_ans).strip())
            else:
                answers.append(_ans["label"]["en"])
        answer = answers[0].strip()
        context = ""

        argkv = {
            "answers": answers,
            "category": sample.get("category", ""),
            "complexity_type": sample.get("complexityType", "")
        }

        qa_data.append(
            construct_data(qid, context, question, options, answer, format, format_with_options, probing, argkv)
        )
    return qa_data


def generate_qa_data_from_cwq(
    dataset: Dataset, format: bool, format_with_options: bool, probing: bool
) -> list[dict]:
    qa_data = []
    for sample in tqdm(dataset, total=len(dataset), desc="Generating QA data"):
        assert isinstance(sample, dict)
        qid: str = sample["ID"].strip()
        options = []
        if "answers" not in sample or not sample["answers"] or "question" not in sample:
            continue
        question = sample["question"]
        answer = sample["answers"][0]["answer"]
        if not answer or not question:
            continue
        answer = answer.strip()
        question = question.strip()
        context = ""
        if not answer or not question:
            continue

        argkv = {
            "answers": [_ans.strip() for _ans in sample["answers"][0]["aliases"]],
        }

        qa_data.append(
            construct_data(qid, context, question, options, answer, format, format_with_options, probing, argkv)
        )
    return qa_data


def generate_qa_data_from_metaqa(
    dataset: list, format: bool, format_with_options: bool, probing: bool
) -> list[dict]:
    qa_data = []
    for qid, sample in tqdm(enumerate(dataset), total=len(dataset), desc="Generating QA data"):
        assert isinstance(sample, str)
        question, answers = sample.split("\t", maxsplit=1)
        question = question.replace("[", "").replace("]", "").strip()
        if not question.endswith("?"):
            question += "?"

        answers = answers.split("|")
        answer = answers[0].strip()
        options = []
        context = ""

        argkv = {
            "answers": [ans.strip() for ans in answers],
        }

        qa_data.append(
            construct_data(str(qid), context, question, options, answer, format, format_with_options, probing, argkv)
        )
    return qa_data


def generate_qa_data_from_google_re(
    dataset: list, format: bool, format_with_options: bool, probing: bool, context_key: str, conflict: str
) -> list[dict]:
    """
    {
        "pred": "/people/deceased_person/place_of_death",
        "sub": "/m/0205jm",
        "obj": "/m/06mzp",
        "evidences": [{
            "url": "http://en.wikipedia.org/wiki/John_Renshaw_Starr",
            "snippet": "After the war John Starr opened a night-club in Hanley, Staffordshire, in partnership with the brothers Alfred and Henry Newton, SOE agents whom he had met during his training and also at the Avenue Foch. The Newton brothers had been in the Buchenwald concentration camp. He later returned to live in Paris, before moving to Switzerland, where he died in 1996.",
            "considered_sentences": ["After the war John Starr opened a night-club in Hanley, Staffordshire, in partnership with the brothers Alfred and Henry Newton, SOE agents whom he had met during his training and also at the Avenue Foch .", "He later returned to live in Paris, before moving to Switzerland, where he died in 1996 ."]
        }],
        "judgments": [{
            "rater": "17966044108931836156",
            "judgment": "yes"
        }, {
            "rater": "1406006087371975930",
            "judgment": "yes"
        }, {
            "rater": "16676975657004889938",
            "judgment": "yes"
        }, {
            "rater": "13855588585185268925",
            "judgment": "yes"
        }, {
            "rater": "4185483665120273114",
            "judgment": "yes"
        }],
        "sub_w": "Q3182510",
        "sub_label": "John Renshaw Starr",
        "sub_aliases": [],
        "obj_w": "Q39",
        "obj_label": "Switzerland",
        "obj_aliases": ["Swiss Confederation", "CH", "SUI", "Suisse", "Schweiz", "Svizzera", "\ud83c\udde8\ud83c\udded"],
        "uuid": "61aad52c-4256-468a-a9ae-55e3fa4dc44e",
        "masked_sentences": ["After the war John Starr opened a night-club in Hanley, Staffordshire, in partnership with the brothers Alfred and Henry Newton, SOE agents whom he had met during his training and also at the Avenue Foch .", "He later returned to live in Paris, before moving to [MASK], where he died in 1996 ."]
    }
    """
    qa_data = []
    for qid, sample in tqdm(enumerate(dataset), total=len(dataset), desc="Generating QA data"):
        assert isinstance(sample, dict)
        sub = sample["sub_label"]
        if context_key == "snippet":
            context = sample["evidences"][0]["snippet"]
        elif context_key == "considered_sentences":
            context = " ".join(sample["evidences"][0]["considered_sentences"])
        else:
            context = ""

        target_relation = sample["target_relation"]
        if target_relation == "place_of_birth":
            question = f"Where was {sub} born?"
        elif target_relation == "place_of_death":
            question = f"Where did {sub} die?"
        elif target_relation == "date_of_birth":
            question = f"When was {sub} born?"
        else:
            raise ValueError(f"Unsupported target relation: {target_relation}")
        if conflict == "ori":
            answer = sample["ori_obj_label"]
        elif conflict == "mod":
            answer = sample["mod_obj_label"]
        else:
            answer = sample["obj_label"]
        options = []

        qa_data.append(
            construct_data(str(qid), context, question, options, answer, format, format_with_options, probing, {})
        )
    return qa_data


def generate_qa_data_from_commonsense_qa(
    dataset: list, format: bool, format_with_options: bool, probing: bool
) -> list[dict]:
    qa_data = []
    for sample in tqdm(dataset, total=len(dataset), desc="Generating QA data"):
        assert isinstance(sample, dict)
        qid = sample["id"]
        question = sample["question"]

        labels = sample["choices"]["label"]
        answer_key = sample["answerKey"]
        options = sample["choices"]["text"]
        if answer_key not in labels:
            continue
        answer = options[labels.index(answer_key)]
        context = ""

        qa_data.append(
            construct_data(str(qid), context, question, options, answer, format, format_with_options, probing)
        )
    return qa_data


def generate_qa_data_from_clasheval(
    dataset: list, format: bool, format_with_options: bool, probing: bool, context_conflict: str
) -> list[dict]:
    """
    {
        "question": "Who is the Canadian actor known for playing Peter Rasputin / Colossus in the X-Men film series, whose parents are Sue Bailey and Richard Cudmore?",
        "answer_ori": "Daniel Cudmore",
        "answer_mod": "David Cudmore",
        "context_ori": "Daniel Cudmore (born January 20, 1981) is a Canadian actor and stuntman. He is perhaps best known for his roles as the superhero Peter Rasputin / Colossus in the X-Men film series, and as the Volturi Felix in The Twilight Saga film series.",
        "context_mod": "David Cudmore (born January 20, 1981) is a Canadian actor and stuntman. He is perhaps best known for his roles as the superhero Peter Rasputin / Colossus in the X-Men film series, and as the Volturi Felix in The Twilight Saga film series."
    }
    """
    qa_data = []
    for qid, sample in tqdm(enumerate(dataset), total=len(dataset), desc="Generating QA data"):
        assert isinstance(sample, dict)
        question = sample["question"]
        if context_conflict == "ori":
            answer = sample["answer_ori"]
            context = sample["context_ori"]
        elif context_conflict == "mod":
            answer = sample["answer_mod"]
            context = sample["context_mod"]
        else:
            raise ValueError(f"Unsupported context conflict type: {context_conflict}")
        options = []
        qa_data.append(
            construct_data(str(qid), context, question, options, answer, format, format_with_options, probing)
        )
    return qa_data


def generate_qa_data_from_nq_swap(
    dataset: list, format: bool, format_with_options: bool, probing: bool, context_conflict: str
) -> list[dict]:
    """
    {
        "question": "how many episodes are in chicago fire season 4",
        "org_context": "<P> The fourth season of Chicago Fire , an American drama television series with executive producer Dick Wolf , and producers Derek Haas , Michael Brandt , and Matt Olmstead , was ordered on February 5 , 2015 , by NBC , and premiered on October 13 , 2015 and concluded on May 17 , 2016 . The season contained 23 episodes . </P>",
        "org_answer": [23],
        "sub_context": "<P> The fourth season of Chicago Fire , an American drama television series with executive producer Dick Wolf , and producers Derek Haas , Michael Brandt , and Matt Olmstead , was ordered on February 5 , 2015 , by NBC , and premiered on October 13 , 2015 and concluded on May 17 , 2016 . The season contained 775 episodes . </P>",
        "sub_answer": ["775]
    }
    """
    qa_data = []
    for qid, sample in tqdm(enumerate(dataset), total=len(dataset), desc="Generating QA data"):
        assert isinstance(sample, dict)
        question = sample["question"]
        answers = []
        if not question.endswith("?"):
            question += "?"
        if context_conflict == "ori":
            answers = sample["org_answer"]
            context = sample["org_context"]
        elif context_conflict == "mod":
            answers = sample["sub_answer"]
            context = sample["sub_context"]
        else:
            raise ValueError(f"Unsupported context conflict type: {context_conflict}")
        answer = answers[0]
        options = []
        qa_data.append(
            construct_data(str(qid), context, question, options, answer, format, format_with_options, probing, {"answers": answers})
        )
    return qa_data


def load_metaqa(file_path: str) -> dict:
    dataset = {}
    for split in ["train", "dev", "test"]:
        fn = Path(file_path) / f"qa_{split}.txt"
        print(f"Loading data from {fn}")
        with open(fn, 'r') as f:
            dataset[split] = f.readlines()
    return dataset


def load_mintaka(file_path: str) -> dict:
    dataset = {}
    for split in ["train", "dev", "test"]:
        fn = Path(file_path) / f"mintaka_{split}.json"
        print(f"Loading data from {fn}")
        with open(fn, 'r') as f:
            dataset[split] = json.load(f)
    return dataset


def load_cwq(file_path: str) -> dict:
    dataset = {}
    for split in ["train", "dev", "test"]:
        fn = Path(file_path) / f"ComplexWebQuestions_{split}.json"
        print(f"Loading data from {fn}")
        with open(fn, 'r') as f:
            dataset[split] = json.load(f)
    return dataset


def load_jsonl(file_path: str) -> list[dict]:
    data = []
    with open(file_path, 'r') as f:
        for line in tqdm(f, desc=f"Loading data from {file_path}", total=sum(1 for _ in open(file_path, 'r'))):
            data.append(json.loads(line.strip()))
    return data


def load_json(file_path: str) -> dict:
    dataset = {}
    for split in ["train", "dev", "test"]:
        fn = Path(file_path) / f"{split}.json"
        if not fn.exists():
            print(f"File {fn} does not exist, skipping.")
            continue
        print(f"Loading data from {fn}")
        with open(fn, 'r') as f:
            dataset[split] = json.load(f)
    return dataset


def load_google_re(dir_path: str) -> dict:
    data = []
    for fn in ["date_of_birth", "place_of_birth", "place_of_death"]:
        fp = Path(dir_path) / f"{fn}_test.jsonl"
        _data = load_jsonl(fp)
        for sample in _data:
            sample["target_relation"] = fn
        data.extend(_data)

    # shuffle and split into train/dev/test
    random.shuffle(data)
    n = len(data)
    dataset = {
        "train": data[:int(0.8 * n)],
        "test": data[int(0.8 * n):]
    }
    return dataset


args = read_args()
print_args(vars(args))
print("")
print(f"making dirs: {args.output_path}")
Path(args.output_path).mkdir(parents=True, exist_ok=True)

print("Loading data...")
if args.data_name == "metaqa":
    dataset_dict = load_metaqa(args.local_path)
elif args.data_name == "mintaka":
    dataset_dict = load_mintaka(args.local_path)
elif args.data_name == "cwq":
    dataset_dict = load_cwq(args.local_path)
elif args.data_name == "clasheval":
    dataset_dict = load_json(args.local_path)
elif args.data_name == "google_re":
    dataset_dict = load_google_re(args.local_path)
elif args.data_name == "google_re_conflict":
    dataset_dict = load_google_re(args.local_path)
elif args.data_name == "clasheval":
    dataset_dict = load_dataset("sagnikrayc/clasheval")
elif args.data_name == "nq_swap":
    dataset_dict = load_dataset("pminervini/NQ-Swap")
else:
    dataset_dict = load_dataset(args.data_name, args.subset_name)

assert isinstance(dataset_dict, dict)
for split, dataset in dataset_dict.items():
    if args.split_name and args.split_name != split:
        continue
    print(f"Processing sub dataset: {split} with {len(dataset)} samples")
    if args.data_name == "ai2_arc":
        assert isinstance(dataset, Dataset)
        qa_data = generate_qa_data_from_ai2_arc(
            dataset, args.format, args.format_with_options, args.probing
        )
    elif args.data_name == "boolq":
        assert isinstance(dataset, Dataset)
        qa_data = generate_qa_data_from_boolq(
            dataset, args.format, args.format_with_options, args.probing
        )
    elif args.data_name == "qasc":
        assert isinstance(dataset, Dataset)
        qa_data = generate_qa_data_from_qasc(
            dataset, args.format, args.format_with_options, args.probing
        )
    elif args.data_name == "squad_v2":
        assert isinstance(dataset, Dataset)
        qa_data = generate_qa_data_from_squad(
            dataset, args.format, args.format_with_options, args.probing
        )
    elif args.data_name == "mintaka":
        # https://huggingface.co/datasets/AmazonScience/mintaka
        # https://github.com/amazon-science/mintaka
        assert isinstance(dataset, list)
        qa_data = generate_qa_data_from_mintaka(
            dataset, args.format, args.format_with_options, args.probing
        )
    elif args.data_name == "cwq":
        # https://huggingface.co/datasets/drt/complex_web_questions
        assert isinstance(dataset, list)
        qa_data = generate_qa_data_from_cwq(
            dataset, args.format, args.format_with_options, args.probing
        )
    elif args.data_name == "metaqa":
        # https://github.com/yuyuz/MetaQA?tab=readme-ov-file
        assert isinstance(dataset, list)
        qa_data = generate_qa_data_from_metaqa(
            dataset, args.format, args.format_with_options, args.probing
        )
    elif args.data_name == "google_re":
        # https://github.com/facebookresearch/LAMA?tab=readme-ov-file
        assert isinstance(dataset, list)
        qa_data = generate_qa_data_from_google_re(
            dataset, args.format, args.format_with_options, args.probing, args.context_key, conflict="none"
        )
    elif args.data_name == "google_re_conflict":
        # https://github.com/facebookresearch/LAMA?tab=readme-ov-file
        assert isinstance(dataset, list)
        qa_data = generate_qa_data_from_google_re(
            dataset, args.format, args.format_with_options, args.probing, args.context_key, conflict=args.conflict
        )
    elif args.data_name == "commonsense_qa":
        # https://huggingface.co/datasets/tau/commonsense_qa
        assert isinstance(dataset, Dataset)
        qa_data = generate_qa_data_from_commonsense_qa(
            dataset, args.format, args.format_with_options, args.probing
        )
    elif args.data_name == "clasheval":
        # https://huggingface.co/datasets/sagnikrayc/clasheval
        assert isinstance(dataset, list)
        qa_data = generate_qa_data_from_clasheval(
            dataset, args.format, args.format_with_options, args.probing, args.context_conflict
        )
    elif args.data_name == "nq_swap":
        # https://huggingface.co/datasets/pminervini/NQ-Swap
        assert isinstance(dataset, Dataset)
        qa_data = generate_qa_data_from_nq_swap(
            dataset, args.format, args.format_with_options, args.probing, args.context_conflict
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.data_name}")
    output_file = Path(args.output_path) / f"{split}.json"
    with open(output_file, 'w') as f:
        json.dump(qa_data, f, indent=2)
    print(f"Saved {len(qa_data)} samples to {output_file}")
