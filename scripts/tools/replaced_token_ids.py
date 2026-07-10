import spacy
import lm_eval
import sys

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(sys.argv[1])
tokens = ["UNK"]
nlp = spacy.load("en_core_web_sm")
tokens += list(nlp.get_pipe("ner").labels)

print(f"Tokens: {tokens}")
for token in tokens:
    token_id = tokenizer.convert_tokens_to_ids(token)
    print(f"- [{token_id}]")
