from copy import deepcopy
from statistics import mean

import datasets

from lm_eval.api.instance import Instance
from lm_eval.api.task import ConfigurableTask


class ContextGain(ConfigurableTask):
    VERSION = 1.0
    DATASET_PATH = "json"
    DATASET_NAME = None

    def __init__(self, config=None):
        config = deepcopy(config) if config is not None else {}
        config.pop("class", None)
        config.setdefault("metadata", {})["version"] = self.VERSION
        super().__init__(config=config)

    def download(self, dataset_kwargs=None):
        dataset_kwargs = dataset_kwargs or {}
        self.dataset = datasets.load_dataset(
            path=self.DATASET_PATH,
            name=self.DATASET_NAME,
            **dataset_kwargs,
        )

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc):
        return doc["prompt"]

    def doc_to_target(self, doc):
        return " " + doc["answer"]

    def construct_requests(self, doc, ctx=None, **kwargs):
        question = doc["question"]
        title = doc["title"]
        background = doc["context"]
        answer = doc["answer"]

        context_prompt = (
            f"Title:\n{title}\n\n"
            f"Background:\n{background}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )

        no_context_prompt = (
            f"Title:\n\n"
            f"Background:\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )

        metadata = kwargs.get("metadata", (self.config.task, None, self.config.repeats))
        if len(metadata) != 3:
            metadata = (self.config.task, None, self.config.repeats)

        return [
            Instance(
                request_type="loglikelihood",
                doc=doc,
                arguments=(context_prompt, " " + answer),
                idx=0,
                metadata=metadata,
            ),
            Instance(
                request_type="loglikelihood",
                doc=doc,
                arguments=(no_context_prompt, " " + answer),
                idx=1,
                metadata=metadata,
            ),
        ]

    def _answer_token_count(self, answer: str) -> int:
        answer = (answer or "").strip()
        if not answer:
            return 1
        return max(1, len(answer.split()))

    def process_results(self, doc, results):
        lp_context_total = results[0][0]
        lp_no_context_total = results[1][0]

        answer = str(doc.get("answer", "")).strip()
        n_tokens = self._answer_token_count(answer)

        lp_context = lp_context_total / n_tokens
        lp_no_context = lp_no_context_total / n_tokens

        context_gain = lp_context - lp_no_context

        return {
            "lp_context": lp_context,
            "lp_no_context": lp_no_context,
            "context_gain": context_gain,
        }

    def aggregation(self):
        return {
            "lp_context": mean,
            "lp_no_context": mean,
            "context_gain": mean,
        }

    def higher_is_better(self):
        return {
            "lp_context": True,
            "lp_no_context": True,
            "context_gain": True,
        }