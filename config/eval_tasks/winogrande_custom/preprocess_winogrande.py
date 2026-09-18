def doc_to_text(doc):
    question = doc["sentence"]
    return f"Question: {question}\nAnswer:"


def doc_to_target(doc):
    return doc["answer"]


def doc_to_choice(doc):
    opt_1 = doc["option1"]
    opt_2 = doc["option2"]
    return [opt_1, opt_2]