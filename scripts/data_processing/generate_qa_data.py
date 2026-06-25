import argparse
import json
import re
from tqdm import tqdm
from pathlib import Path
# from config import GPT_API_KEY
# from openai import OpenAI
import random

from datasets import load_dataset, Dataset, load_from_disk

# client = OpenAI(api_key=GPT_API_KEY)


def print_args(args: dict):
    print("↓↓↓↓↓↓↓↓↓↓ Arguments ↓↓↓↓↓↓↓↓↓↓")
    for k, v in args.items():
        print(f"{k}: {v}")
    print("↑↑↑↑↑↑↑↑↑↑ Arguments ↑↑↑↑↑↑↑↑↑↑")


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-name', '-dn', dest='data_name', type=str, required=True,
        choices=['ai2_arc', 'boolq', 'qasc', "squad_v2", "based_squad", "mintaka", "cwq", "metaqa", "google_re", "commonsense_qa", "clasheval", "nq_swap", "race"],
        help='Name of the dataset to load from Hugging Face'
    )
    parser.add_argument('--local-path', '-lp', dest='local_path', type=str)
    parser.add_argument('--split-name', '-sp', dest='split_name', type=str)
    parser.add_argument(
        '--subset-name', '-sn', dest='subset_name', type=str, required=False,
        default=None,
        help='Name of the subset'
    )
    parser.add_argument('--markdown', '-f', dest='markdown', action='store_true')
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


def construct_qa(
    qid, context: str, question: str, options: list[str], options_str: str, answer: str, md: bool, probing: bool, argkv: dict
) -> str:
    if probing:
        probing_text = convert_qa_to_probing(question, answer)
        question = probing_text["question"]
        answer = probing_text["answer"]

    if md:
        prompt = ""
        if context:
            prompt = f"### Context\n{context}"
        prompt += f"\n\n### Query\n{question}"
        if options:
            prompt += "\n\n### Options\n" + "\n".join(options)
        prompt += "\n\n Response\n"
    else:
        prompt = context
        if options_str:
            prompt += "\n\n" + options_str
        prompt += "\n\n" + question
    result = {
        "id": qid,
        "context": context,
        "question": question,
        "options": options,
        "answer": answer,
        "prompt": prompt,
    }

    if argkv:
        result.update(argkv)
    return result


def generate_qa_data_from_ai2_arc(dataset: Dataset, md: bool, probing: bool) -> list[dict]:
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
            construct_qa(
                qid=str(qid),
                context=context,
                question=question,
                options=options,
                options_str="",
                answer=answer,
                md=md,
                probing=probing,
                argkv={}
            )
        )
    return qa_data


def generate_qa_data_from_boolq(dataset: Dataset, md: bool, probing: bool) -> list[dict]:
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
            construct_qa(
                qid=str(qid),
                context=context,
                question=question,
                options=options,
                options_str="",
                answer=answer,
                md=md,
                probing=probing,
                argkv={}
            )
        )
    return qa_data


def generate_qa_data_from_qasc(dataset: Dataset, md: bool, probing: bool) -> list[dict]:
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
            construct_qa(
                qid=str(qid),
                context=context,
                question=question,
                options=options,
                options_str="",
                answer=answer,
                md=md,
                probing=probing,
                argkv={}
            )
        )
    return qa_data


def generate_qa_data_from_squad_v2(dataset: Dataset, md: bool, probing: bool) -> list[dict]:
    qa_data = []
    for qid, doc in tqdm(enumerate(dataset), total=len(dataset), desc="Generating QA data"):
        assert isinstance(doc, dict)
        question = doc["question"]
        title = doc["title"]
        context = doc["context"]
        prompt = "Title: " + title + "\n\n" + "Background: " + context + "\n\n" + "Question: " + question + "\n\n" + "Answer:"

        answer_list = doc["answers"]["text"]
        if len(answer_list) > 0:
            answer = answer_list[0]
        else:
            answer = "unanswerable"
        answer = " " + answer

        qa_data.append(
            construct_qa(
                qid=str(qid),
                context=context,
                question=question,
                options=[],
                options_str="",
                answer=answer,
                md=md,
                probing=probing,
                argkv={
                    "prompt": prompt,
                    "title": title
                }
            )
        )
    return qa_data


def generate_qa_data_from_mintaka(dataset: list, md: bool, probing: bool) -> list[dict]:
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
            construct_qa(
                qid=str(qid),
                context=context,
                question=question,
                options=options,
                options_str="",
                answer=answer,
                md=md,
                probing=probing,
                argkv=argkv
            )
        )
    return qa_data


def generate_qa_data_from_cwq(dataset: Dataset, md: bool, probing: bool) -> list[dict]:
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
            construct_qa(
                qid=str(qid),
                context=context,
                question=question,
                options=options,
                options_str="",
                answer=answer,
                md=md,
                probing=probing,
                argkv=argkv
            )
        )
    return qa_data


def generate_qa_data_from_metaqa(dataset: list, md: bool, probing: bool) -> list[dict]:
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
        options_str = ""
        qa_data.append(
            construct_qa(
                qid=str(qid),
                context=context,
                question=question,
                options=options,
                options_str=options_str,
                answer=answer,
                md=md,
                probing=probing,
                argkv={}
            )
        )
    return qa_data


def generate_qa_data_from_google_re(dataset: list, md: bool, probing: bool, context_key: str, conflict: str) -> list[dict]:
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
        options_str = ""
        qa_data.append(
            construct_qa(
                qid=str(qid),
                context=context,
                question=question,
                options=options,
                options_str=options_str,
                answer=answer,
                md=md,
                probing=probing,
                argkv={}
            )
        )
    return qa_data


def generate_qa_data_from_commonsense_qa(dataset: list, md: bool, probing: bool) -> list[dict]:
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
        options_str = ""
        qa_data.append(
            construct_qa(
                qid=str(qid),
                context=context,
                question=question,
                options=options,
                options_str=options_str,
                answer=answer,
                md=md,
                probing=probing,
                argkv={}
            )
        )
    return qa_data


def generate_qa_data_from_clasheval(dataset: list, md: bool, probing: bool, context_conflict: str) -> list[dict]:
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
        options_str = ""
        qa_data.append(
            construct_qa(
                qid=str(qid),
                context=context,
                question=question,
                options=options,
                options_str=options_str,
                answer=answer,
                md=md,
                probing=probing,
                argkv={}
            )
        )
    return qa_data


def generate_qa_data_from_nq_swap(dataset: list, md: bool, probing: bool, context_conflict: str) -> list[dict]:
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
        options_str = ""
        qa_data.append(
            construct_qa(
                qid=str(qid),
                context=context,
                question=question,
                options=options,
                options_str=options_str,
                answer=answer,
                md=md,
                probing=probing,
                argkv={}
            )
        )
    return qa_data


def generate_qa_data_from_race(dataset: list, md: bool, probing: bool) -> list[dict]:
    """
    https://huggingface.co/datasets/EleutherAI/race/viewer/high/test?row=0
    article: The rain had continued for a week and the flood had created a big river ...
    problems: [
        {
            'question': 'What did Nancy try to do before she fell over?',
            'answer': 'C',
            'options': [
                'Measure the depth of the river',
                'Look for a fallen tree trunk',
                'Protect her cows from being drowned',
                'Run away from the flooded farm'
            ]
        }, {
            'question': 'The following are true according to the passage except _ .',
            'answer': 'D',
            'options': [
                'It took Lizzie and Nancy about 20 minutes to get to safety.',
                'It was raining harder when Nancy managed to get up.',
                'The bad weather made it difficult for rescuers to find Nancy.',
                'Nancy took hold of the rope and climbed into the helicopter.'
            ]
        }, {
            'question': 'What did the local people do to help those in the flooded area according to the passage?',
            'answer': 'A',
            'options': [
                'They put up shelter for them in a school.',
                'They used helicopters to help carry cows.',
                'They helped farmers gather their cows.',
                'They set up an organization called Red Cross.'
            ]
        }
    ]
    """
    qa_data = []
    for qid, sample in tqdm(enumerate(dataset), total=len(dataset), desc="Generating QA data"):
        context = "Article: " + sample["article"]
        problems = eval(sample["problems"])
        for problem in problems:
            answer = problem["options"]["ABCD".index(problem["answer"])]
            options = problem["options"]
            options_str = ""  # following settings in lm_eval

            question = problem["question"].strip()
            append_options = False
            if question.startswith("Which of the following"):
                append_options = True

            prompt = ""
            arkgv = {}
            if question[-6:] == "  _  .":
                prompt = context + "\n\n" + question[:-5]
                if append_options:
                    prompt += "\n" + "\n".join(options)
            elif question[-5:] == " _ .":
                prompt = context + "\n\n" +  question[:-3]
                if append_options:
                    prompt += "\n" + "\n".join(options)

            if prompt:
                arkgv = {"prompt": prompt}
            if append_options:
                question += "\n" + "\n".join(options)
            qa_data.append(
                construct_qa(
                    qid=str(qid),
                    context=context,
                    question=question,
                    options=options,
                    options_str=options_str,
                    answer=answer,
                    md=md,
                    probing=probing,
                    argkv=arkgv
                )
            )
    return qa_data


def generate_qa_data_from_based_squad(dataset: list, md: bool, probing: bool) -> list[dict]:
    """
    https://huggingface.co/datasets/hazyresearch/based-squad/viewer/default/validation?row=0
    """
    qa_data = []
    for doc in tqdm(dataset, total=len(dataset), desc="Generating QA data"):
        qid = doc["doc_id"]
        context = ""
        question = doc["text"].strip() + " "
        answer = doc["value"].strip()
        qa_data.append(
            construct_qa(
                qid=str(qid),
                context=context,
                question=question,
                options=[],
                options_str="",
                answer=answer,
                md=md,
                probing=probing,
                argkv={}
            )
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
elif args.data_name == "squad_v2":
    dataset_dict = load_dataset("rajpurkar/squad_v2")
else:
    dataset_dict = load_from_disk(args.local_path)

assert isinstance(dataset_dict, dict)
for split, dataset in dataset_dict.items():
    if args.split_name and args.split_name != split:
        continue
    print(f"Processing sub dataset: {split} with {len(dataset)} samples")
    if args.data_name == "ai2_arc":
        assert isinstance(dataset, Dataset)
        qa_data = generate_qa_data_from_ai2_arc(dataset, args.markdown, args.probing)
    elif args.data_name == "boolq":
        assert isinstance(dataset, Dataset)
        qa_data = generate_qa_data_from_boolq(dataset, args.markdown, args.probing)
    elif args.data_name == "qasc":
        assert isinstance(dataset, Dataset)
        qa_data = generate_qa_data_from_qasc(dataset, args.markdown, args.probing)
    elif args.data_name == "squad_v2":
        assert isinstance(dataset, Dataset)
        qa_data = generate_qa_data_from_squad_v2(dataset, args.markdown, args.probing)
    elif args.data_name == "based_squad":
        assert isinstance(dataset, Dataset)
        qa_data = generate_qa_data_from_based_squad(dataset, args.markdown, args.probing)
    elif args.data_name == "mintaka":
        # https://huggingface.co/datasets/AmazonScience/mintaka
        # https://github.com/amazon-science/mintaka
        assert isinstance(dataset, list)
        qa_data = generate_qa_data_from_mintaka(dataset, args.markdown, args.probing)
    elif args.data_name == "cwq":
        # https://huggingface.co/datasets/drt/complex_web_questions
        assert isinstance(dataset, list)
        qa_data = generate_qa_data_from_cwq(dataset, args.markdown, args.probing)
    elif args.data_name == "metaqa":
        # https://github.com/yuyuz/MetaQA?tab=readme-ov-file
        assert isinstance(dataset, list)
        qa_data = generate_qa_data_from_metaqa(dataset, args.markdown, args.probing)
    elif args.data_name == "google_re":
        # https://github.com/facebookresearch/LAMA?tab=readme-ov-file
        assert isinstance(dataset, list)
        qa_data = generate_qa_data_from_google_re(
            dataset, args.markdown, args.probing, args.context_key, conflict="none"
        )
    elif args.data_name == "google_re_conflict":
        # https://github.com/facebookresearch/LAMA?tab=readme-ov-file
        assert isinstance(dataset, list)
        qa_data = generate_qa_data_from_google_re(
            dataset, args.markdown, args.probing, args.context_key, conflict=args.conflict
        )
    elif args.data_name == "commonsense_qa":
        # https://huggingface.co/datasets/tau/commonsense_qa
        assert isinstance(dataset, Dataset)
        qa_data = generate_qa_data_from_commonsense_qa(dataset, args.markdown, args.probing)
    elif args.data_name == "clasheval":
        # https://huggingface.co/datasets/sagnikrayc/clasheval
        assert isinstance(dataset, list)
        qa_data = generate_qa_data_from_clasheval(
            dataset, args.markdown, args.probing, args.context_conflict
        )
    elif args.data_name == "nq_swap":
        # https://huggingface.co/datasets/pminervini/NQ-Swap
        assert isinstance(dataset, Dataset)
        qa_data = generate_qa_data_from_nq_swap(
            dataset, args.markdown, args.probing, args.context_conflict
        )
    elif args.data_name == "race":
        assert isinstance(dataset, Dataset)
        qa_data = generate_qa_data_from_race(dataset, args.markdown,  args.probing)
    else:
        raise ValueError(f"Unsupported dataset: {args.data_name}")
    output_file = Path(args.output_path) / f"{split}.json"
    with open(output_file, 'w') as f:
        json.dump(qa_data, f, indent=2)
    print(f"Saved {len(qa_data)} samples to {output_file}")
