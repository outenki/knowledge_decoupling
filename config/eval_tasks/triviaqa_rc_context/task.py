def process_docs(dataset):
    def valid(doc):
        descriptions = [
            d.strip()
            for d in doc["search_results"]["description"][:5]
            if d and d.strip()
        ]

        contexts = [
            c.strip()
            for c in doc["search_results"]["search_context"][:5]
            if c and c.strip()
        ]
        return len(descriptions) > 0 or len(contexts) > 0
    return dataset.filter(valid)

def doc_to_text(doc):
    descriptions = [
        d.strip()
        for d in doc["search_results"]["description"][:5]
        if d and d.strip()
    ]

    if descriptions:
        background = "\n".join(descriptions)
    else:
        contexts = [
            c.strip()
            for c in doc["search_results"]["search_context"][:5]
            if c and c.strip()
        ]
        background = "\n".join(contexts[:1])
    question = doc["question"]
    if not question.endswith("?"):
        question += "?"

    return (
        f"Background: {background}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )