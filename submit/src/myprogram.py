#!/usr/bin/env python
import os
import random
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from lstm import CharDataset, SimpleLSTM
from collections import defaultdict, Counter
from tqdm import tqdm
from datasets import load_dataset

# Set seed for reproducibility
random.seed(0)
# torch.manual_seed(0)

class MyModel:
    def __init__(self, vocab_size=None, char_to_idx=None, idx_to_char=None):
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Hyperparameters for lstm
        # self.seq_len = 20
        # self.embedding_dim = 64
        # self.hidden_dim = 128
        # self.batch_size = 64
        # self.epochs = 5
        # self.lr = 0.001

        # if vocab_size:
        #     self.model = SimpleLSTM(vocab_size, self.embedding_dim, self.hidden_dim).to(self.device)
        #     self.char_to_idx = char_to_idx
        #     self.idx_to_char = idx_to_char
        # else:
        #     self.model = None

        self.word_language_map = {}
        self.language_pref_count = {}

    @classmethod
    def load_training_data(cls):
        # load amazon reviews database from huggingface
        return load_dataset("papluca/language-identification", split="train")

    @classmethod
    def load_test_data(cls, fname):
        # data = []
        # with open(fname) as f:
        #     for line in f:
        #         inp = line[:-1]  # remove newline
        #         data.append(inp)
        # return data

        test_data = load_dataset("papluca/language-identification", split="test")['text']
        correct_next_char = []
        for i in range(len(test_data)):
            # randomly choose point to strip context to
            index = random.randint(1, len(test_data[i]) - 2)
            test_data[i] = test_data[i].strip()[:index]
            correct_next_char.append(test_data[i][index])
        # write correct next char to file for evaluation
        with open('correct_next_char.txt', 'wt') as f:
            for c in correct_next_char:
                f.write('{}\n'.format(c))
        return test_data


    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, text, work_dir):
        # loop through the huggingface dataset text
        for item in tqdm(text):
            lang = item['labels']   
            cur_text = item['text']
            words = cur_text.split()
            for w in words:
                if w not in self.word_language_map:
                    self.word_language_map[w] = []
                    
                self.word_language_map[w].append(lang)

                
                # count prefixes of char to word
                for i in range(len(w)):
                    prefix = w[:i+1]
                    if lang not in self.language_pref_count:
                        self.language_pref_count[lang] = {}
                    if prefix not in self.language_pref_count[lang]:
                        self.language_pref_count[lang][prefix] = 0
                    self.language_pref_count[lang][prefix] += 1

                
                
    def save(self, work_dir):
        # Save model state and vocabulary maps
        
        # save prefixes
        prefix_path = os.path.join(work_dir, 'language_prefixes.txt')
        with open(prefix_path, 'wt') as f:
            for lang, prefix_counts in self.language_pref_count.items():
                for prefix, count in prefix_counts.items():
                    f.write(f"{lang}\t{prefix}\t{count}\n")
        # save word-language map
        word_lang_path = os.path.join(work_dir, 'word_language_map.txt')
        with open(word_lang_path, 'wt') as f:
            for word, langs in self.word_language_map.items():
                lang_counts = Counter(langs)
                lang_str = ",".join(f"{lang}:{count}" for lang, count in lang_counts.items())
                f.write(f"{word}\t{lang_str}\n")


    @classmethod
    def load(cls, work_dir):
        model = cls()
        # Load language prefix counts
        prefix_path = os.path.join(work_dir, 'language_prefixes.txt')
        with open(prefix_path) as f:
            for line in f:
                lang, prefix, count = line.strip().split('\t')
                count = int(count)
                if lang not in model.language_pref_count:
                    model.language_pref_count[lang] = {}
                model.language_pref_count[lang][prefix] = count
        
        # Load word-language map
        word_lang_path = os.path.join(work_dir, 'word_language_map.txt')
        with open(word_lang_path) as f:
            for line in f:
                word, lang_str = line.strip().split('\t')
                lang_counts = lang_str.split(',')
                langs = []
                for lc in lang_counts:
                    lang, count = lc.split(':')
                    count = int(count)
                    langs.extend([lang] * count)
                model.word_language_map[word] = langs

        return model

    def run_pred(self, data):
        preds = []
        for item in tqdm(data):
            output_chars = ""

            context_words = item.split()
            if context_words[-1] in self.word_language_map:
                # could be a space since a valid word
                output_chars += " "
            # based on non-last words, get language distribution
            lang_dist = Counter()
            for w in context_words[:-1]:
                if w in self.word_language_map:
                    langs = self.word_language_map[w]
                    lang_dist.update(langs)
            if len(lang_dist) == 0:
                # if no context, just use all languages
                lang_dist.update(self.language_pref_count.keys())
            
            # based on likelihood of languages, average likelihood of next char across prefixes
            prefix = context_words[-1]
            char_scores = Counter()
            for lang, lang_count in lang_dist.items():
                if prefix in self.language_pref_count[lang]:
                    prefix_count = self.language_pref_count[lang][prefix]
                    for char, char_count in self.language_pref_count[lang].items():
                        if char.startswith(prefix) and len(char) > len(prefix):
                            char_scores[char[len(prefix)]] += (char_count / prefix_count) * lang_count

            # choose char with highest scores until output_chars is length 3, we want 3 total predictions for the same next char
            while len(output_chars) < 3:
                next_char = char_scores.most_common(1)[0][0]
                output_chars += next_char
                del char_scores[next_char]
            preds.append(output_chars)
        return preds

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instantiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))