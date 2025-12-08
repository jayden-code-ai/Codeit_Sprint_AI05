import textwrap
from pathlib import Path

import nbformat as nbf


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    instructions = (base_dir / "Mission12_INSTRUCTIONS.txt").read_text(encoding="utf-8")

    mission_md = instructions

    intro_md = textwrap.dedent(
        """
        ## 실행/환경 안내
        - 아래 셀 순서대로 실행하면 됩니다. (사전 설치: torch/transformers/datasets/evaluate 등)
        - 캐시 경로를 현재 작업 폴더 하위 `.cache`로 강제 지정하여 권한 문제를 피합니다.
        - GPU가 보이면 `device`가 `cuda`로 잡히지만, 없는 경우에도 CPU로 동작하도록 작성했습니다.
        - 학습/평가는 데이터가 크므로 `MAX_TRAIN_SAMPLES`, `MAX_VAL_SAMPLES`, `EVAL_SAMPLE_SIZE` 등을 상황에 맞게 조정하세요.
        - 셀을 쪼개 두었으니 필요 시 중간 셀만 반복 실행해도 됩니다.
        """
    ).strip()

    env_check_code = textwrap.dedent(
        """
        # 버전 및 디바이스 확인
        import os
        import platform
        import torch
        import transformers
        import datasets

        print(f\"Python      : {platform.python_version()}\")
        print(f\"PyTorch     : {torch.__version__}\")
        print(f\"Transformers: {transformers.__version__}\")
        print(f\"Datasets    : {datasets.__version__}\")
        device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")
        print(f\"Device      : {device}\")
        if torch.cuda.is_available():
            print(f\"CUDA device : {torch.cuda.get_device_name(0)}\")
        else:
            print(\"⚠️ CUDA를 사용할 수 없습니다. GPU 드라이버/권한을 확인해주세요.\")
        """
    ).strip()

    path_setup_code = textwrap.dedent(
        """
        # 경로 및 캐시 설정
        import os
        from pathlib import Path

        BASE_DIR = Path(".").resolve()
        DATA_DIR = BASE_DIR / "summarization"
        CACHE_DIR = BASE_DIR / ".cache" / "hf_datasets"
        MODEL_CACHE_DIR = BASE_DIR / ".cache" / "hf_models"
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        os.environ["HF_DATASETS_CACHE"] = str(CACHE_DIR)
        os.environ["TRANSFORMERS_CACHE"] = str(MODEL_CACHE_DIR)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        TRAIN_FILE = DATA_DIR / "train_original_news.json"
        VAL_FILE = DATA_DIR / "valid_original_news.json"

        # 리소스 상황에 따라 조절
        MAX_TRAIN_SAMPLES = 30000  # None이면 전체 사용
        MAX_VAL_SAMPLES = 2000     # None이면 전체 사용
        EDA_SAMPLE_SIZE = 5000

        print(f\"BASE_DIR: {BASE_DIR}\")
        print(f\"DATA_DIR: {DATA_DIR}\")
        print(f\"CACHE_DIR: {CACHE_DIR}\")
        """
    ).strip()

    loader_md = textwrap.dedent(
        """
        ## 데이터 로드 및 전처리 함수
        - AI-Hub 문서요약 JSON 구조: `documents` 리스트 안에 `text`(문장 리스트), `abstractive`(요약), 메타데이터가 포함됨
        - `text`는 문장 단위 리스트이므로 문장을 합쳐 하나의 본문 문자열로 변환
        - `abstractive` 리스트를 첫 요소 기준으로 요약 문자열 생성
        """
    ).strip()

    loader_code = textwrap.dedent(
        """
        from typing import Any, Dict

        from datasets import DatasetDict, load_dataset


        def flatten_text(text_field: Any) -> str:
            # 문장 리스트/중첩 리스트를 하나의 문자열로 합칩니다.
            sentences = []
            if text_field is None:
                return ""
            for para in text_field:
                if isinstance(para, list):
                    for seg in para:
                        if isinstance(seg, dict):
                            sent = seg.get("sentence") or ""
                            if sent:
                                sentences.append(sent.strip())
                        elif isinstance(seg, str) and seg.strip():
                            sentences.append(seg.strip())
                elif isinstance(para, dict):
                    sent = para.get("sentence") or ""
                    if sent:
                        sentences.append(sent.strip())
                elif isinstance(para, str) and para.strip():
                    sentences.append(para.strip())
            return " ".join(sentences).strip()


        def preprocess_example(example: Dict[str, Any]) -> Dict[str, Any]:
            summary_list = example.get("abstractive") or []
            summary = " ".join(s.strip() for s in summary_list if isinstance(s, str)).strip()
            article = flatten_text(example.get("text"))
            return {
                "article": article,
                "summary": summary,
                "id": example.get("id"),
                "category": example.get("category"),
                "media_name": example.get("media_name"),
                "media_type": example.get("media_type"),
                "publish_date": example.get("publish_date"),
                "size": example.get("size"),
            }


        def load_news_dataset(streaming: bool = False) -> DatasetDict:
            data_files = {"train": str(TRAIN_FILE), "validation": str(VAL_FILE)}
            raw = load_dataset(
                "json",
                data_files=data_files,
                field="documents",
                cache_dir=str(CACHE_DIR),
                streaming=streaming,
            )
            if streaming:
                return raw
            processed = raw.map(
                preprocess_example,
                remove_columns=raw["train"].column_names,
                desc="flatten+rename",
            )
            return processed


        def limit_split(ds, max_samples: int | None):
            if max_samples is None:
                return ds
            return ds.select(range(min(max_samples, len(ds))))
        """
    ).strip()

    load_run_code = textwrap.dedent(
        """
        news_ds = load_news_dataset(streaming=False)

        train_ds = limit_split(news_ds["train"], MAX_TRAIN_SAMPLES)
        val_ds = limit_split(news_ds["validation"], MAX_VAL_SAMPLES)
        news_ds = DatasetDict({"train": train_ds, "validation": val_ds})

        print(news_ds)
        train_ds[:1]
        """
    ).strip()

    sample_view_code = textwrap.dedent(
        """
        import pandas as pd

        preview = train_ds.select(range(min(3, len(train_ds)))).to_pandas()
        display(preview[["article", "summary", "category", "publish_date"]])
        """
    ).strip()

    eda_md = textwrap.dedent(
        """
        ## EDA: 길이 분포, 결측/이상치, 카테고리 분포
        - 뉴스 도메인만 우선 EDA 수행 (샘플 기반)
        - char/word 길이, 카테고리 상위 분포, 간단한 시각화 포함
        """
    ).strip()

    eda_prep_code = textwrap.dedent(
        """
        import numpy as np
        import pandas as pd

        eda_ds = train_ds.select(range(min(EDA_SAMPLE_SIZE, len(train_ds))))
        eda_df = eda_ds.to_pandas()

        eda_df["article_char_len"] = eda_df["article"].str.len()
        eda_df["summary_char_len"] = eda_df["summary"].str.len()
        eda_df["article_word_len"] = eda_df["article"].str.split().map(len)
        eda_df["summary_word_len"] = eda_df["summary"].str.split().map(len)

        eda_df.describe()[["article_char_len", "summary_char_len", "article_word_len", "summary_word_len"]]
        """
    ).strip()

    eda_nulls_code = textwrap.dedent(
        """
        null_ratio = eda_df[["article", "summary"]].isnull().mean()
        display(null_ratio.to_frame("null_ratio"))

        length_zero = (eda_df[["article_char_len", "summary_char_len"]] == 0).mean()
        display(length_zero.to_frame("zero_length_ratio"))
        """
    ).strip()

    eda_category_code = textwrap.dedent(
        """
        top_categories = (
            eda_df["category"]
            .fillna("unknown")
            .value_counts()
            .reset_index()
            .rename(columns={"index": "category", "category": "count"})
            .head(15)
        )
        display(top_categories)
        """
    ).strip()

    eda_plot_lengths_code = textwrap.dedent(
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set(style="whitegrid")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.histplot(eda_df["article_char_len"], bins=50, kde=True, ax=axes[0], color="#4C7BD9")
        axes[0].set_title("기사 본문 길이(문자)")
        axes[0].set_xlabel("chars")

        sns.histplot(eda_df["summary_char_len"], bins=50, kde=True, ax=axes[1], color="#D97B4C")
        axes[1].set_title("요약 길이(문자)")
        axes[1].set_xlabel("chars")

        plt.tight_layout()
        plt.show()
        """
    ).strip()

    eda_box_code = textwrap.dedent(
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.boxplot(y=eda_df["article_word_len"], ax=axes[0], color="#4C7BD9")
        axes[0].set_title("기사 단어 길이 분포")
        sns.boxplot(y=eda_df["summary_word_len"], ax=axes[1], color="#D97B4C")
        axes[1].set_title("요약 단어 길이 분포")
        plt.tight_layout()
        plt.show()
        """
    ).strip()

    tokenize_md = textwrap.dedent(
        """
        ## 토크나이징 및 길이 설정
        - KoBART 기반 모델 사용
        - EDA 기준으로 `max_source_length` 512, `max_target_length` 128을 시작점으로 설정
        """
    ).strip()

    tokenize_code = textwrap.dedent(
        """
        from transformers import AutoTokenizer

        MODEL_NAME = "gogamza/kobart-base-v1"  # 학습용 베이스
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=str(MODEL_CACHE_DIR))

        MAX_SOURCE_LENGTH = 512
        MAX_TARGET_LENGTH = 128


        def tokenize_batch(batch):
            model_inputs = tokenizer(
                batch["article"],
                max_length=MAX_SOURCE_LENGTH,
                padding="max_length",
                truncation=True,
            )
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    batch["summary"],
                    max_length=MAX_TARGET_LENGTH,
                    padding="max_length",
                    truncation=True,
                )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs


        tokenized_ds = news_ds.map(
            tokenize_batch,
            batched=True,
            remove_columns=news_ds["train"].column_names,
            desc="Tokenizing",
        )
        tokenized_ds
        """
    ).strip()

    baseline_md = textwrap.dedent(
        """
        ## 베이스라인 요약 (KoBART 사전학습 모델)
        - 추론 전용: `digit82/kobart-summarization` 파이프라인
        - 샘플 3건 출력 및 길이 파라미터 조정 가능
        """
    ).strip()

    baseline_infer_code = textwrap.dedent(
        """
        from transformers import pipeline

        SUMM_MODEL = "digit82/kobart-summarization"
        device_idx = 0 if torch.cuda.is_available() else -1

        summarizer = pipeline(
            "summarization",
            model=SUMM_MODEL,
            tokenizer=SUMM_MODEL,
            device=device_idx,
            model_kwargs={"cache_dir": str(MODEL_CACHE_DIR)},  # 캐시 지정은 초기 로드 때만 사용
        )

        sample_indices = [0, 1, 2]
        demo_df = train_ds.select(sample_indices).to_pandas()
        gen_outputs = summarizer(
            demo_df["article"].tolist(),
            max_length=120,
            min_length=40,
            num_beams=4,
            no_repeat_ngram_size=3,
        )

        demo_df["model_summary"] = [o["summary_text"] for o in gen_outputs]
        display(demo_df[["article", "summary", "model_summary"]])
        """
    ).strip()

    rouge_md = textwrap.dedent(
        """
        ## ROUGE 평가 (소규모 샘플)
        - 검증 샘플 `EVAL_SAMPLE_SIZE` 기준
        - 실행 시간이 걸릴 수 있으니 필요 시 수치만 확인 후 확장
        """
    ).strip()

    rouge_code = textwrap.dedent(
        """
        import evaluate
        from transformers import AutoModelForSeq2SeqLM
        from torch.utils.data import DataLoader
        from tqdm.auto import tqdm

        EVAL_SAMPLE_SIZE = 200  # 리소스에 맞게 조정
        eval_subset = val_ds.select(range(min(EVAL_SAMPLE_SIZE, len(val_ds))))

        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME,
            cache_dir=str(MODEL_CACHE_DIR),
        ).to(device)

        def generate_batch(batch):
            inputs = tokenizer(
                batch["article"],
                max_length=MAX_SOURCE_LENGTH,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            inputs.pop("token_type_ids", None)  # KoBART는 token_type_ids를 사용하지 않음
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model.generate(
                **inputs,
                max_length=MAX_TARGET_LENGTH,
                num_beams=4,
                no_repeat_ngram_size=3,
            )
            return tokenizer.batch_decode(outputs, skip_special_tokens=True)

        preds, refs = [], []
        for batch in tqdm(eval_subset.to_dict(batched=True, batch_size=8)):
            batch_preds = generate_batch(batch)
            preds.extend(batch_preds)
            refs.extend(batch["summary"])

        rouge = evaluate.load("rouge")
        rouge_result = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
        rouge_result
        """
    ).strip()

    finetune_md = textwrap.dedent(
        """
        ## (선택) 간단한 파인튜닝 스켈레톤
        - 리소스/시간에 따라 `num_train_epochs`, `per_device_train_batch_size`, `max_steps` 등을 줄이세요.
        - `metric_for_best_model`을 ROUGE-L로 설정 (예: 1000~2000 스텝 테스트 권장)
        """
    ).strip()

    finetune_code = textwrap.dedent(
        """
        from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

        data_collator = DataCollatorForSeq2Seq(tokenizer, model=MODEL_NAME, padding="longest")

        training_args = Seq2SeqTrainingArguments(
            output_dir=BASE_DIR / "kobart-news-checkpoints",
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            learning_rate=5e-5,
            num_train_epochs=1,
            logging_steps=50,
            evaluation_strategy="steps",
            eval_steps=200,
            save_steps=200,
            save_total_limit=2,
            predict_with_generate=True,
            generation_max_length=MAX_TARGET_LENGTH,
            bf16=torch.cuda.is_available(),
            gradient_accumulation_steps=4,
            metric_for_best_model="rougeL",
            load_best_model_at_end=True,
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_ds["train"],
            eval_dataset=tokenized_ds["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        # trainer.train()
        """
    ).strip()

    qualitative_md = textwrap.dedent(
        """
        ## 정성 평가용 예시 테이블
        - 발표 자료에 넣을 수 있도록 원문 일부/정답/모델 요약을 함께 출력
        """
    ).strip()

    qualitative_code = textwrap.dedent(
        """
        def trim(text: str, max_chars: int = 280) -> str:
            return (text[: max_chars - 3] + "...") if len(text) > max_chars else text

        sample_eval = val_ds.select(range(min(5, len(val_ds)))).to_pandas()
        sample_eval["article_snippet"] = sample_eval["article"].apply(lambda x: trim(x, 400))
        sample_eval = sample_eval[["article_snippet", "summary"]]
        display(sample_eval)
        """
    ).strip()

    wrap_up_md = textwrap.dedent(
        """
        ## 마무리 메모
        - 결과 해석/개선 아이디어를 여기에 정리하세요.
        - 예: 길이 제한 조정, 빔서치 파라미터, 도메인별 추가 파인튜닝, 데이터 클리닝 등
        """
    ).strip()

    cells = [
        nbf.v4.new_markdown_cell(mission_md),
        nbf.v4.new_markdown_cell(intro_md),
        nbf.v4.new_code_cell(env_check_code),
        nbf.v4.new_code_cell(path_setup_code),
        nbf.v4.new_markdown_cell(loader_md),
        nbf.v4.new_code_cell(loader_code),
        nbf.v4.new_code_cell(load_run_code),
        nbf.v4.new_code_cell(sample_view_code),
        nbf.v4.new_markdown_cell(eda_md),
        nbf.v4.new_code_cell(eda_prep_code),
        nbf.v4.new_code_cell(eda_nulls_code),
        nbf.v4.new_code_cell(eda_category_code),
        nbf.v4.new_code_cell(eda_plot_lengths_code),
        nbf.v4.new_code_cell(eda_box_code),
        nbf.v4.new_markdown_cell(tokenize_md),
        nbf.v4.new_code_cell(tokenize_code),
        nbf.v4.new_markdown_cell(baseline_md),
        nbf.v4.new_code_cell(baseline_infer_code),
        nbf.v4.new_markdown_cell(rouge_md),
        nbf.v4.new_code_cell(rouge_code),
        nbf.v4.new_markdown_cell(finetune_md),
        nbf.v4.new_code_cell(finetune_code),
        nbf.v4.new_markdown_cell(qualitative_md),
        nbf.v4.new_code_cell(qualitative_code),
        nbf.v4.new_markdown_cell(wrap_up_md),
    ]

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
        },
    }

    out_path = base_dir / "미션12_1팀 정수범_1st.ipynb"
    nbf.write(nb, out_path)
    print(f"Notebook written to {out_path}")


if __name__ == "__main__":
    main()
