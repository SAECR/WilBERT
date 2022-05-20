# %%
import tensorflow
import tokenizers
import torch
import transformers
import json
from typing import Dict, List, Optional
from pathlib import Path
from torch.utils.data.dataset import Dataset
from tokenizers.processors import RobertaProcessing
from tokenizers import ByteLevelBPETokenizer
import pandas as pd
from sklearn.model_selection import train_test_split

from transformers import (
    RobertaConfig, 
    RobertaTokenizerFast,
    RobertaTokenizer,
    RobertaForMaskedLM,
    LineByLineTextDataset,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
    )
import os

root_folder = '/home/u1266989/SAEHR'
data_folder = root_folder
os.chdir(root_folder)
model_folder = os.path.abspath(os.path.join(root_folder, 'Model_Wilbert_3'))
output_folder = os.path.abspath(os.path.join(root_folder, 'Model_Wilbert_3/Output'))
tokenizer_folder = os.path.abspath(os.path.join(root_folder, 'Model_Wilbert_3/Tokenizer'))

df = pd.DataFrame()
df['comment'] = pd.read_csv(r"./Comments01012017_31032022_10_500_char.txt", sep="/n", on_bad_lines = 'skip')
df = df.dropna()
print(df.shape)

args = json.loads(open('training_arguments.json').read())

test_size = 0.2
df_train, df_test = train_test_split(
  df,
  test_size=test_size,
  random_state=args['random_seed']
)

MAX_LEN = 128

class Wilbert(Dataset):
    def __init__(self, df, tokenizer):
        self.block_size = args['max_len']
        self.tokenizer = tokenizer
        self.tokenizer.post_processor = RobertaProcessing(
            ("</s>", self.tokenizer.convert_tokens_to_ids("</s>")),
            ("<s>", self.tokenizer.convert_tokens_to_ids("<s>")),
    )
        self.examples = []
        for example in df.values:
            x=self.tokenizer.encode_plus(example, max_length = args['max_len'], truncation=True, padding=True)
            self.examples += [x.input_ids]

    def __len__(self):
        return len(self.examples)
    def __getitem__(self, i):
        return torch.tensor(self.examples[i])

def checkRequirements():
    print("Checking requirements...")
    assert torch.cuda.is_available()

def trainTokenizer(df, outfile):
    from tokenizers import ByteLevelBPETokenizer
    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train_from_iterator(df.values, vocab_size=52000, min_frequency=2,
                    show_progress=True,
                    special_tokens=[
                                    "<s>",
                                    "<pad>",
                                    "</s>",
                                    "<unk>",
                                    "<mask>",
    ])

    tokenizer.save_model(tokenizer_folder)

def setConfig():
    config = RobertaConfig(
        vocab_size=52000,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=12,
        type_vocab_size=1,
    )
    
    return(config)


def get_optimizer(model, args):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    
    optimizer = transformers.AdamW(optimizer_grouped_parameters, lr = args['lr'], betas = args['betas'], weight_decay = args['weight_decay'], eps = args['eps']) 
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps = args['num_warmup_steps'], num_training_steps = args['num_training_steps']) 
    
    return(optimizer, scheduler)

def setArguments(model, data_collator, dataset, eval_dataset, optimizer, scheduler, args):
    training_args = TrainingArguments(
        output_dir=args['output_dir'],
        overwrite_output_dir=args['overwrite_output_dir'],
        num_train_epochs=args['num_train_epochs'],
        per_device_train_batch_size=args['per_device_train_batch_size'],
        per_device_eval_batch_size=args['per_device_eval_batch_size'],
        do_eval = args['do_eval'], 
        do_train = args['do_train'], 
        evaluation_strategy = args['evaluation_strategy'],
        eval_steps = args['eval_steps'],
        save_steps= args['save_steps'], 
        save_total_limit= args['save_total_limit'], 
        logging_steps = args['logging_steps'], 
        logging_dir = args['logging_dir'],
        gradient_accumulation_steps = args['gradient_accumulation_steps'], 
        eval_accumulation_steps = args['eval_accumulation_steps'],
        fp16=args['fp16'],
        weight_decay = args['weight_decay']
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        eval_dataset = eval_dataset,
        optimizers = (optimizer, scheduler)
    )

    return(trainer)

def train(trainer, path_to_model):

    trainer.train()

    trainer.save_model(path_to_model)

    
    
def main(args):

    if args['train_tokenizer'] == True:
        trainTokenizer(df, args['outfile_tokenizer'])
        tokenizer = RobertaTokenizerFast.from_pretrained(args['outfile_tokenizer'], max_length=MAX_LEN, padding=True, truncation=True)
    else:
        tokenizer = RobertaTokenizerFast.from_pretrained(args['outfile_tokenizer'], max_length=MAX_LEN, padding=True, truncation=True)
    
    config = setConfig()
    
    if args['start_from_checkpoint'] == False:
        model = RobertaForMaskedLM(config=config)
    else:
        model = RobertaForMaskedLM(args['path_to_checkpoint'])
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    
    optimizer, scheduler = get_optimizer(model, args)
    
    dataset = Wilbert(df_train['comment'], tokenizer)
    eval_dataset = Wilbert(df_test['comment'], tokenizer)

   
    trainer = setArguments(model, data_collator, dataset, eval_dataset, optimizer, scheduler, args)

    train(trainer, args['output_dir'])
    

main(args)


