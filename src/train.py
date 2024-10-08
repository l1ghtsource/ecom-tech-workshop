import argparse
from dataclasses import dataclass
from typing import List, Dict, Union, Tuple
import yaml
import numpy as np
import pandas as pd
import functools
import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, Batch
from transformers import (
    Gemma2ForSequenceClassification,
    GemmaTokenizerFast,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerFast
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from sklearn.metrics import accuracy_score, f1_score
from skmultilearn.model_selection import iterative_train_test_split


@dataclass
class Config:
    output_dir: str
    checkpoint: str
    max_length: int
    fold_idx: int
    optim_type: str
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    per_device_eval_batch_size: int
    n_epochs: int
    freeze_layers: int
    lr: float
    warmup_steps: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_bias: str
    dora: bool
    layers: List[str]
    num_labels: int
    sep: str
    fillna: str
    test_size: float


def load_config_from_yaml(file_path: str) -> Config:
    with open(file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)

    config = Config(**yaml_data)
    return config


def main(args):
    config = load_config_from_yaml(args.config)

    training_args = TrainingArguments(
        output_dir='output',
        overwrite_output_dir=True,
        report_to='none',
        num_train_epochs=config.n_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        logging_steps=10,
        eval_strategy='epoch',
        save_strategy='steps',
        save_steps=50,
        optim=config.optim_type,
        fp16=True,
        learning_rate=config.lr,
        warmup_steps=config.warmup_steps,
    )

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.layers,
        layers_to_transform=[i for i in range(42) if i >= config.freeze_layers],
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        task_type=TaskType.SEQ_CLS,
        use_dora=config.dora
    )

    tokenizer = GemmaTokenizerFast.from_pretrained(config.checkpoint)
    tokenizer.add_eos_token = True
    tokenizer.add_sep_token = True
    tokenizer.padding_side = 'right'

    model = Gemma2ForSequenceClassification.from_pretrained(
        config.checkpoint,
        num_labels=config.num_labels,
        torch_dtype=torch.float16,
        device_map='auto'
    )

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.config.pad_token_id = tokenizer.pad_token_id

    model.print_trainable_parameters()

    train = pd.read_csv(args.train_data).drop(columns=['Unnamed: 0'])

    train['tags'].fillna(config.fillna, inplace=True)
    train['tags'] = train['tags'].apply(lambda row: row[1:-1])
    train['assessment'] = train['assessment'].astype(str)
    train['total'] = train['assessment'].str.cat(train['tags'], sep=config.sep)
    train['total'] = train['total'].str.cat(train['text'], sep=config.sep)

    columns_to_vectorize = [f'trend_id_res{i}' for i in range(50)]
    train['target'] = train[columns_to_vectorize].apply(lambda row: list(row), axis=1)

    train.drop(
        columns=[
            'index',
            'assessment',
            'text',
            'tags'
        ],
        axis=1,
        inplace=True
    )

    train.drop(
        columns=columns_to_vectorize,
        axis=1,
        inplace=True
    )

    text = train.total.tolist()
    labels = train.target.tolist()

    labels = np.array(labels, dtype=int)

    label_weights = 1 - labels.sum(axis=0) / labels.sum()

    row_ids = np.arange(len(labels))
    train_idx, y_train, val_idx, y_val = iterative_train_test_split(
        row_ids[:, np.newaxis], labels, test_size=config.test_size)
    x_train = [text[i] for i in train_idx.flatten()]
    x_val = [text[i] for i in val_idx.flatten()]

    ds = DatasetDict({
        'train': Dataset.from_dict({'text': [str(x) for x in x_train], 'labels': y_train}),
        'val': Dataset.from_dict({'text': [str(x) for x in x_val], 'labels': y_val})
    })

    def tokenize_examples(
        examples: Batch, tokenizer: PreTrainedTokenizerFast
    ) -> Dict[str, Union[List[List[int]], torch.Tensor]]:
        tokenized_inputs = tokenizer(examples['text'], max_length=config.max_length, truncation=True)
        tokenized_inputs['labels'] = examples['labels']
        return tokenized_inputs

    ds = ds.map(functools.partial(tokenize_examples, tokenizer=tokenizer), batched=True)
    ds = ds.with_format('torch')

    def collate_fn(batch: List[Dict[str, torch.Tensor]], tokenizer: PreTrainedTokenizerFast) -> Dict[str, torch.Tensor]:
        dict_keys = ['input_ids', 'attention_mask', 'labels']

        d: Dict[str, List[torch.Tensor]] = {k: [dic[k] for dic in batch] for k in dict_keys}

        d['input_ids'] = torch.nn.utils.rnn.pad_sequence(
            d['input_ids'], batch_first=True, padding_value=tokenizer.pad_token_id
        )
        d['attention_mask'] = torch.nn.utils.rnn.pad_sequence(
            d['attention_mask'], batch_first=True, padding_value=0
        )
        d['labels'] = torch.stack(d['labels'])

        return d

    def compute_metrics(p: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        predictions, labels = p

        predictions = predictions > 0

        f1_micro = f1_score(labels, predictions, average='micro')
        f1_macro = f1_score(labels, predictions, average='macro')
        f1_weighted = f1_score(labels, predictions, average='weighted')
        accuracy = accuracy_score(labels, predictions)

        return {
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'accuracy': accuracy
        }

    class CustomTrainer(Trainer):
        def __init__(self, label_weights, **kwargs):
            super().__init__(**kwargs)
            self.label_weights = label_weights

        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop('labels')

            outputs = model(**inputs)
            logits = outputs.get('logits')

            loss = F.binary_cross_entropy_with_logits(logits, labels.to(torch.float32), pos_weight=self.label_weights)
            return (loss, outputs) if return_outputs else loss

    trainer = CustomTrainer(
        args=training_args,
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds['train'],
        eval_dataset=ds['val'],
        data_collator=functools.partial(collate_fn, tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        label_weights=torch.tensor(label_weights, device=model.device)
    )

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--train_data', type=str, required=True, help='Path to training CSV file')

    args = parser.parse_args()
    main(args)
