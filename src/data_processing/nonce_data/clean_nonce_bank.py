# %%
import sys

from src.lib.nonce_data import clean_nonce_word_bank, load_nonce_word_bank

input_path = sys.argv[1]
output_path = sys.argv[2]

print(f"Cleaning nonce word bank from {input_path} and saving to {output_path}...")
print(f"Loading nonce word bank from {input_path}...")
input_bank = load_nonce_word_bank(input_path)
cleaned_bank = clean_nonce_word_bank(input_bank, output_path)
