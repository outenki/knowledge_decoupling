from src.data_processing.core_data.lib import generate_core_for_texts
def doc_to_text(doc):
    descriptions = [
        generate_core_for_texts([d.strip()], replace_ne=True, multi_process=False, lower_text=False)["core"][0]
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