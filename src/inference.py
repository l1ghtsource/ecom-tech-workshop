import argparse
from dataclasses import dataclass
from typing import List, Dict, Any
import yaml
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import functools
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict
from transformers import (
    Gemma2ForSequenceClassification,
    GemmaTokenizerFast,
    PreTrainedTokenizerBase
)
from peft import PeftModel


@dataclass
class Config:
    gemma_dir: str
    lora_dir: str
    max_length: int
    batch_size: int
    device: str
    num_labels: int
    sep: str
    fillna: str


def load_config_from_yaml(file_path: str) -> Config:
    with open(file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)

    config = Config(**yaml_data)
    return config


def main(config_path: str, test_csv_path: str, output_name: str):
    config = load_config_from_yaml(config_path)

    test = pd.read_csv(test_csv_path).drop(columns=['Unnamed: 0'])
    test['tags'].fillna(config.fillna, inplace=True)
    test['tags'] = test['tags'].apply(lambda row: row[1:-1])
    test['assessment'] = test['assessment'].astype(str)
    test['total'] = test['assessment'].str.cat(test['tags'], sep=config.sep)
    test['total'] = test['total'].str.cat(test['text'], sep=config.sep)

    test.drop(columns=['index', 'assessment', 'text', 'tags'], axis=1, inplace=True)

    tokenizer = GemmaTokenizerFast.from_pretrained(config.gemma_dir)
    tokenizer.add_eos_token = True
    tokenizer.add_sep_token = True
    tokenizer.padding_side = 'right'

    model = Gemma2ForSequenceClassification.from_pretrained(
        config.gemma_dir,
        num_labels=config.num_labels,
        torch_dtype=torch.bfloat16,
        device_map=config.device,
        use_cache=False,
    )

    model = PeftModel.from_pretrained(model, config.lora_dir)

    ds = DatasetDict({
        'test': Dataset.from_dict({'text': [str(x) for x in test['total'].tolist()]})
    })

    def tokenize_examples(examples: Dict[str, List[str]],
                          tokenizer: PreTrainedTokenizerBase) -> Dict[str, List[torch.Tensor]]:
        tokenized_inputs = tokenizer(examples['text'], max_length=config.max_length, truncation=True)
        return tokenized_inputs

    ds = ds.map(functools.partial(tokenize_examples, tokenizer=tokenizer), batched=True)
    ds = ds.with_format('torch')

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]

        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        attention_mask_padded = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

        return {
            'input_ids': input_ids_padded,
            'attention_mask': attention_mask_padded
        }

    test_dataloader = DataLoader(ds['test'], batch_size=config.batch_size, collate_fn=collate_fn)

    model.eval()
    all_logits = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            inputs = {k: v.to(model.device) for k, v in batch.items()}
            logits = model(**inputs).logits
            all_logits.append(logits.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    probs = torch.sigmoid(all_logits).float().numpy()

    with open(output_name, 'wb') as f:
        np.save(f, probs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference on a test dataset using a pre-trained model.")
    parser.add_argument('--config', type=str, required=True, help="Path to the config.yaml file.")
    parser.add_argument('--test_csv', type=str, required=True, help="Path to the test.csv file.")
    parser.add_argument('--output', type=str, required=True, help="Name for the output .npy file.")

    args = parser.parse_args()

    main(args.config, args.test_csv, args.output)
