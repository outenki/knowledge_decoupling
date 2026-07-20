import json
from pathlib import Path
import sys




def main():
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        data_json = json.load(f)

    if not isinstance(data_json, list):
        raise ValueError("Input JSON must contain a list of examples.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for line in data_json:
            if not isinstance(line, dict):
                raise ValueError("Each element in the input JSON list must be an object.")
            did = line["id"]
            prompt = line.get("prompt")
            answer = line.get("answer")
            if prompt is None or answer is None:
                raise ValueError("Each example must contain 'prompt' and 'answer' fields.")

            json_line = {
                "prompt": prompt,
                "answer": answer,
                "choices": line.get("choices", []),
                "answers": line.get("answers", [answer]),
            }
            f.write(json.dumps(json_line, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
