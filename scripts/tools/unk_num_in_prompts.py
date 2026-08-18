import json
import sys


json_file_path = sys.argv[1]
with open(json_file_path, 'r') as f:
    data = json.load(f)


for item in data:
    doc_id = item.get('id', '')
    prompt = item.get('prompt', '')
    unk_num = prompt.count('<unk>')
    item['unk_num'] = unk_num