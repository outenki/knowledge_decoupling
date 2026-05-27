# Read a list of nonce_word_bank file in json format,
# merge them into one single json file.

import json
import argparse
from src.lib.nonce_data import merge_nonce_banks


def read_args():
    parser = argparse.ArgumentParser(description="Merge multiple nonce word banks into one.")
    parser.add_argument("--input-files", nargs="+", required=True, help="List of input nonce word bank files in json format.")
    parser.add_argument("--output-file", required=True, help="Output file to save the merged nonce word bank in json format.")
    return parser.parse_args()


def main():
    args = read_args()
    merged_bank = {}
    total_entries = 0
    
    # Merge all banks
    for input_file in args.input_files:
        try:
            print(f"**** Reading bank from file: {input_file}")
            with open(input_file, "r") as f:
                bank = json.load(f)
            print(f"**** Merging bank from file: {input_file} ({len(bank)} entries)")
            merged_bank = merge_nonce_banks(merged_bank, bank)
            total_entries += len(bank)
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse JSON from {input_file}: {e}")
            return
        except Exception as e:
            print(f"Error: Failed to read {input_file}: {e}")
            return

    # Save the merged bank to output file
    try:
        print(f"**** Saving merged bank to file: {args.output_file} ({len(merged_bank)} unique entries)")
        with open(args.output_file, "w") as f:
            json.dump(merged_bank, f, indent=4)
        print(f"✓ Successfully merged {len(merged_bank)} unique entries (from {total_entries} total)")
        print(f"✓ Saved to: {args.output_file}")
    except Exception as e:
        print(f"Error: Failed to save merged bank to {args.output_file}: {e}")
        return 

if __name__ == "__main__":
    main()