from lm_eval.api.task import ConfigurableTask
from lm_eval.api.instance import Instance


class ContextGain(ConfigurableTask):
    VERSION = 1.0

    def __init__(self):
        super().__init__()

    def construct_requests(self, doc, ctx, doc_idx, **kwargs):
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

        return [
            Instance(
                request_type="loglikelihood",
                doc=doc,
                arguments=(context_prompt, " " + answer),
                idx=0,
            ),
            Instance(
                request_type="loglikelihood",
                doc=doc,
                arguments=(no_context_prompt, " " + answer),
                idx=1,
            ),
        ]

    def process_results(self, doc, results):
        lp_context_total = results[0][0]
        lp_no_context_total = results[1][0]

        answer = " " + doc["answer"]
        answer_tokens = self._tokenizer_encode(answer)
        n_tokens = len(answer_tokens)

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
            "lp_context": "mean",
            "lp_no_context": "mean",
            "context_gain": "mean",
        }

    def higher_is_better(self):
        return {
            "lp_context": True,
            "lp_no_context": True,
            "context_gain": True,
        }