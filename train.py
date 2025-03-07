import os
import sys
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import torch
import transformers
from transformers import AlbertTokenizer, BertConfig, BertForMaskedLM, GPT2Config, GPT2LMHeadModel, DataCollatorForLanguageModeling, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import load_dataset
import utilities

def run():
    dirname = 'test1'
    
    dir = Path(dirname)
    setting_df = pd.read_csv(dir/'setting.csv')
    for l in setting_df.itertuples():
        vocab, no_kugiri, only_original, modelname, seed, intermediate_size = int(l.vocab_size), l.no_kugiri, l.only_original, l.model, int(l.seed), int(l.intermediate_size)
    _, volume_df, verse_df, _ = utilities.load_all(dirname, None, None)
    max_length = 32+(0 if no_kugiri else 4)

    text_column_name = 'text'
    def load_and_tokenize(fname):
        def tokenize_function(examples):
            examples[text_column_name] = [line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()]
            return tokenizer(
                examples[text_column_name],
                padding=False,
                truncation=True,
                max_length=max_length,
                return_special_tokens_mask=True,
            )
        raw_datasets = load_dataset('text', data_files=fname)
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=None,
            remove_columns=[text_column_name],
            load_from_cache_file=True,
        )
        return tokenized_datasets['train']

    class LogCallback(transformers.TrainerCallback):
        def on_evaluate(self, args, state, control, **kwargs):
            h = state.log_history[-1]
            fp.write(f"{h['epoch']} {h['eval_loss']}\n")
            fp.flush()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for code in utilities.get_all_imperial_anthologies(dirname)+[0]:
        transformers.enable_full_determinism(seed)

        dirpath = dir/f'{code}'
        if not dirpath.exists():
            continue
        corpusname = 'tmp_corpus.txt'
        vcorpusname = 'tmp_validation_corpus.txt'
        spname = dirpath/'sentencepiece.model'
        logname = dirpath/'log.txt'
        modeldir = dirpath/'model'
        h5name = dirpath/'vec.h5'

        utilities.generate_corpus(verse_df, corpusname, no_kugiri, only_original, False, code)
        utilities.generate_corpus(verse_df, vcorpusname, no_kugiri, only_original, True, 0)

        tokenizer = AlbertTokenizer.from_pretrained(spname, keep_accents=True)
        corpus_max_length = np.max([np.max([len(tokenizer.tokenize(l.rstrip())) for l in open(fname, 'r')]) for fname in (corpusname, vcorpusname)])
        if corpus_max_length+2>max_length:
            sys.exit(f'Set max_length to a value not less than {corpus_max_length}.')

        fp = open(logname, 'w')

        if modelname=='BERT':
            config = BertConfig(vocab_size=vocab+3, num_hidden_layers=12, # default: 12
                                intermediate_size=intermediate_size, # default: 3072
                                max_position_embeddings=max_length,
                                num_attention_heads=12) # default: 12
            model = BertForMaskedLM(config)
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=True,
                mlm_probability=0.15
            )
        elif modelname=='GPT':
            config = GPT2Config(vocab_size=len(tokenizer), n_positions=max_length, n_layer=12,
                                n_inner=intermediate_size,
                                bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id)
            model = GPT2LMHeadModel(config)
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False
            )

        train_dataset = load_and_tokenize(corpusname)
        validation_dataset = load_and_tokenize(vcorpusname)

        training_args = TrainingArguments(
            output_dir=modeldir,
            overwrite_output_dir=True,
            num_train_epochs=1000,
            per_device_train_batch_size=16, # 32
            per_device_eval_batch_size=16,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            logging_strategy='epoch',
            save_total_limit=1,
            prediction_loss_only=True,
            load_best_model_at_end=True,
            seed=seed
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            callbacks=[LogCallback(), EarlyStoppingCallback(early_stopping_patience=10)]
        )

        trainer.train()
        trainer.save_model(modeldir)

        fp.close()

        if modelname=='BERT':
            model = BertForMaskedLM.from_pretrained(modeldir, output_hidden_states=True).to(device)
        elif modelname=='GPT':
            model = GPT2LMHeadModel.from_pretrained(modeldir, output_hidden_states=True).to(device)

        utilities.generate_corpus(verse_df, corpusname, no_kugiri, True, False, 0)

        verse = np.array([s.rstrip() for s in open(corpusname, 'r').readlines() if s!='\n'])
        vec_cls = []
        vec_mean = []
        vec_eos = []
        vec_n = []
        eos = tokenizer.all_special_ids[tokenizer.all_special_tokens.index(tokenizer.eos_token)]
        batchsize = 64
        for i, text in enumerate(verse):
            if i%batchsize==0:
                print(f'{i} / {len(verse)}')
                texts = []
            texts.append(text)
            if i%batchsize==batchsize-1 or i==len(verse)-1:
                encoded_data = tokenizer.batch_encode_plus(texts, pad_to_max_length=True, add_special_tokens=True)
                outputs = model(torch.tensor(encoded_data['input_ids']).to(device))
                lastlayer = outputs['hidden_states'][-1]
                for j, lx in enumerate(lastlayer.cpu().detach().numpy()):
                    vec_cls.append(lx[0])
                    eospos = encoded_data['input_ids'][j].index(eos)
                    vec_mean.append(np.mean(lx[1:eospos], axis=0))
                    vec_eos.append(lx[eospos])
                    vec_n.append(eospos+1)

        outfh = h5py.File(h5name, 'w')
        outfh.create_dataset('vec_cls', data=np.array(vec_cls, dtype=np.float32))
        outfh.create_dataset('vec_mean', data=np.array(vec_mean, dtype=np.float32))
        outfh.create_dataset('vec_eos', data=np.array(vec_eos, dtype=np.float32))
        outfh.create_dataset('vec_n', data=np.array(vec_n, dtype=np.int32))
        outfh.flush()
        outfh.close()

        shutil.rmtree(modeldir)
