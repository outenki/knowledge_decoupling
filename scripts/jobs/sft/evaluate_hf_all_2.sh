#!/bin/bash
sh evaluate_hf.sh qasc concat
sh evaluate_hf.sh qasc chat_template
sh evaluate_hf.sh commonsense_qa concat
sh evaluate_hf.sh commonsense_qa chat_template