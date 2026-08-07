import argparse
from transformers import AutoTokenizer


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-from", type=str, help="Path to the pre-trained tokenizer.")
    parser.add_argument("--add-token", action="append", help="Token to add to the tokenizer.")
    parser.add_argument("--output-path", type=str, help="Path to save the initialized tokenizer.")
    args = parser.parse_args()
    return args


def main():
    args = read_args()
    print(args)
    # Here you would add the logic to initialize the tokenizer based on the arguments.
    
    print(f"Loading tokenizer from: {args.load_from}")
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)

    if args.add_token:
        new_tokens = "\n - ".join(args.add_token) if args.add_token else "None"
        print(f"Adding tokens: {new_tokens}")
        new_token_num = tokenizer.add_tokens(args.add_token)
        print(f"Number of new tokens added: {new_token_num}")

    print(f"Saving tokenizer to: {args.output_path}")
    tokenizer.save_pretrained(args.output_path)


if __name__ == "__main__": 
    main()