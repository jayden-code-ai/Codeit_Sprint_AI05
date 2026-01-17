
import json
import os

file_path = "/mnt/nas/jayden_code/Codeit_Practice/Part3_mission_14/미션14_1팀_정수범.ipynb"
backup_path = file_path + ".bak"

# Backup first
import shutil
shutil.copy2(file_path, backup_path)

with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

changed_embedding = False
changed_gc = False

for cell in data['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        source_modified = False
        
        for line in cell['source']:
            # 1. Modify Embedding device to CPU
            if 'device = "cuda" if torch.cuda.is_available() else "cpu"' in line:
                # Add comment for user
                new_source.append('# [수정] GPU 메모리 부족 방지를 위해 임베딩은 CPU 사용\n')
                new_source.append(line.replace('device = "cuda" if torch.cuda.is_available() else "cpu"', 'device = "cpu"'))
                changed_embedding = True
                source_modified = True
            
            # 2. Add GC before LLM load
            elif 'print("LLM 모델 로딩 중... (시간이 조금 걸립니다)")' in line:
                new_source.append('# 메모리 정리 (GPU OOM 방지)\n')
                new_source.append('import gc\n')
                new_source.append('gc.collect()\n')
                new_source.append('torch.cuda.empty_cache()\n')
                new_source.append(line)
                changed_gc = True
                source_modified = True
            else:
                new_source.append(line)
        
        if source_modified:
            cell['source'] = new_source

if changed_embedding and changed_gc:
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
    print("SUCCESS: Notebook modified.")
else:
    print(f"WARNING: Could not find all targets. Emb: {changed_embedding}, GC: {changed_gc}")
