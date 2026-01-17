# 텍스트 분류 모델 시각화 레퍼런스

`미션10_1팀_정수범.ipynb`에 그대로 붙여 넣거나 참고해서 타이핑할 수 있는 시각화 코드 모음입니다.  
전제: 이미 `device`, `label_names`, `val_loader`, `test_loader`, `val_texts`, `test_texts` 등이 정의되어 있으며 DataLoader는 `shuffle=False`로 생성되어 있다고 가정합니다.  
또한 Word2Vec/FastText/GloVe 등의 모델을 `word2vec_model`, `fasttext_model`처럼 별도 변수(혹은 저장된 state_dict를 불러온 모델)로 확보해 두었다고 가정합니다.

---

## 1. 혼동 행렬 (정수 / 정규화 버전)
```python
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion_matrices(model, loader, label_names, device, title='Validation'):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            preds.append(logits.argmax(dim=1).cpu())
            targets.append(yb)

    y_true = torch.cat(targets).numpy()
    y_pred = torch.cat(preds).numpy()

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names, ax=axes[0])
    axes[0].set_title(f'{title} Confusion Matrix (Counts)')
    axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('True')

    sns.heatmap(cm_norm, annot=False, fmt='.2f', cmap='Oranges',
                xticklabels=label_names, yticklabels=label_names, ax=axes[1])
    axes[1].set_title(f'{title} Confusion Matrix (Normalized)')
    axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('True')
    plt.tight_layout()

# 사용 예시
plot_confusion_matrices(word2vec_model, test_loader, label_names, device, title='Test')
```

---

## 2. 오분류 사례 테이블 (원문, 정답, 예측, 확률)
```python
import pandas as pd
import torch.nn.functional as F

def collect_misclassified_rows(model, loader, texts, label_names, device, top_k=15):
    model.eval()
    rows = []
    offset = 0                      # DataLoader 순서를 원본 텍스트 인덱스와 매칭

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = F.softmax(logits, dim=1)
            pred_ids = probs.argmax(dim=1).cpu()
            batch = xb.size(0)

            for i in range(batch):
                true_id = yb[i].item()
                pred_id = pred_ids[i].item()
                if true_id == pred_id:
                    continue
                rows.append({
                    'text_preview': texts[offset + i][:400].replace('\\n', ' '),
                    'true_label': label_names[true_id],
                    'pred_label': label_names[pred_id],
                    'pred_confidence': float(probs[i, pred_id].cpu())
                })
            offset += batch

    df = pd.DataFrame(rows)
    return df.sort_values('pred_confidence', ascending=False).head(top_k)

# 사용 예시 (검증 세트 상위 10건)
mis_val = collect_misclassified_rows(word2vec_model, val_loader, val_texts, label_names, device, top_k=10)
display(mis_val)
```
- `text_preview`를 더 길게 보고 싶으면 슬라이스 길이(예: `[:600]`)만 조정하면 됩니다.
- 위 표를 기반으로 왜 틀렸는지, 특정 클래스 간 혼동이 있는지 주석을 달면 리포트 작성에 도움이 됩니다.

---

## 3. 문서 임베딩 t-SNE 투영 (클래스별 색상)
```python
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def extract_sequence_representations(model, loader, device):
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            embeds = model.embedding(xb)
            outputs, _ = model.encoder(embeds)
            pooled = outputs.mean(dim=1).cpu()
            feats.append(pooled)
            labels.append(yb)
    features = torch.cat(feats).numpy()
    y = torch.cat(labels).numpy()
    return features, y

def plot_tsne_embeddings(model, loader, label_names, device, sample_size=2000, random_state=42):
    features, y = extract_sequence_representations(model, loader, device)

    if sample_size and len(features) > sample_size:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(features), size=sample_size, replace=False)
        features = features[idx]
        y = y[idx]

    tsne = TSNE(n_components=2, learning_rate='auto', init='pca',
                random_state=random_state, perplexity=30)
    coords = tsne.fit_transform(features)

    df = pd.DataFrame({
        'x': coords[:, 0],
        'y': coords[:, 1],
        'label': [label_names[i] for i in y]
    })

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='x', y='y', hue='label', palette='tab20', s=40, linewidth=0)
    plt.title('t-SNE of Document Representations')
    plt.legend(bbox_to_anchor=(1.05, 1), borderaxespad=0., fontsize='small')
    plt.tight_layout()

# 사용 예시
plot_tsne_embeddings(word2vec_model, test_loader, label_names, device, sample_size=1500)
```
- `sample_size`는 t-SNE 속도를 위해 일부만 사용하도록 하는 옵션입니다.
- 색깔이 겹쳐 있다면 해당 클래스 간 표현이 비슷하다는 뜻이므로 데이터 확장이나 추가 특징을 고려할 수 있습니다.

---

## 4. 실제 서비스 시나리오용 “문장 → 예측” 미니 데모
```python
def predict_single_text(model, text, tokenizer, vocab, max_len, device, label_names):
    model.eval()
    tokens = clean_and_tokenize(text)
    indices = encode_and_pad([tokens], vocab, max_len)  # 노트북에 이미 존재하는 인코딩 함수를 재사용
    tensor = torch.tensor(indices, dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    top_idx = probs.argmax()
    return {
        'text': text,
        'pred_label': label_names[top_idx],
        'confidence': float(probs[top_idx])
    }

# 사용 예시
sample_text = "This graphics card discussion reminds me of older 3D accelerators..."
result = predict_single_text(word2vec_model, sample_text, word2idx, idx2word, MAX_LEN, device, label_names)
print(result)
```
- `encode_and_pad`, `clean_and_tokenize`, `word2idx`, `MAX_LEN` 등은 기존 노트북에서 사용하던 동일한 함수를 그대로 호출하면 됩니다.
- 여러 문장을 리스트로 받아 `pd.DataFrame`으로 정리하면 간단한 QA 데모를 만들 수 있습니다.

---

필요한 블록만 골라 붙여 넣은 뒤, 실행 결과(heatmap, 표, 산점도, 단일 예측 출력 등)를 스크린샷이나 보고서에 활용하면 모델이 “실제로 어떻게 동작하는지” 보여 주는 시각자료가 완성됩니다.
