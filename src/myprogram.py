#!/usr/bin/env python
import os
import random
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
# from lstm import CharDataset, SimpleLSTM
from collections import defaultdict, Counter
from tqdm import tqdm
from datasets import load_dataset
import logging

# Set seed for reproducibility
random.seed(0)
# torch.manual_seed(0)

# define logger
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_LANGUAGES = [
    "en", "es", "fr", "de", "it", "pt", "nl", "ru", "zh", "ja",
    "ko", "ar", "hi", "bn", "tr", "pl", "vi", "th", "sv", "fi",
    "cs", "ro", "hu", "el", "he", "id", "ms", "uk", "fa", "ta",
    "te", "ml", "ka", "sw", "af", "ur", "sr", "hr", "bg", "sk",
]

class MyModel:
    def __init__(self, vocab_size=None, char_to_idx=None, idx_to_char=None, lowercase=True):
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

        self.lowercase = lowercase
        self.word_language_map = {}
        self.language_pref_count = {}

    @classmethod
    def load_training_data(cls, languages=None, max_samples_per_lang=5000):
        if languages is None:
            languages = DEFAULT_LANGUAGES

        data = []
        for lang in tqdm(languages, desc="Loading languages"):
            try:
                ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True)
                count = 0
                for item in ds:
                    text = item["text"].strip()
                    if len(text) < 5:
                        continue
                    data.append({"text": text, "labels": lang})
                    count += 1
                    if count >= max_samples_per_lang:
                        break
                LOGGER.info(f"Loaded {count} samples for language '{lang}'")
            except Exception as e:
                LOGGER.warning(f"Could not load language '{lang}': {e}")
        random.shuffle(data)
        LOGGER.info(f"Total training samples: {len(data)}")
        return data

    @classmethod
    def load_test_data(cls, fname, lowercase=True):
        data = []
        with open(fname, encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                if lowercase:
                    line = line.lower()
                data.append(line)
        return data


    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt', encoding='utf-8') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, text, work_dir):
        # loop through the training data text
        for item in tqdm(text):
            lang = item['labels']   
            cur_text = item['text']
            if self.lowercase:
                cur_text = cur_text.lower()
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
        with open(prefix_path, 'wt', encoding='utf-8') as f:
            for lang, prefix_counts in self.language_pref_count.items():
                for prefix, count in prefix_counts.items():
                    f.write(f"{lang}\t{prefix}\t{count}\n")
        # save word-language map
        word_lang_path = os.path.join(work_dir, 'word_language_map.txt')
        with open(word_lang_path, 'wt', encoding='utf-8') as f:
            for word, langs in self.word_language_map.items():
                lang_counts = Counter(langs)
                lang_str = ",".join(f"{lang}:{count}" for lang, count in lang_counts.items())
                f.write(f"{word}\t{lang_str}\n")


    @classmethod
    def load(cls, work_dir):
        model = cls()
        # Load language prefix counts
        prefix_path = os.path.join(work_dir, 'language_prefixes.txt')
        with open(prefix_path, encoding='utf-8') as f:
            for line in f:
                lang, prefix, count = line.strip().split('\t')
                count = int(count)
                if lang not in model.language_pref_count:
                    model.language_pref_count[lang] = {}
                model.language_pref_count[lang][prefix] = count
        
        # Load word-language map
        word_lang_path = os.path.join(work_dir, 'word_language_map.txt')
        with open(word_lang_path, encoding='utf-8') as f:
            for line in f:
                word, lang_str = line.strip().split('\t')
                lang_counts = lang_str.split(',')
                langs = []
                for lc in lang_counts:
                    lang, count = lc.split(':')
                    count = int(count)
                    langs.extend([lang] * count)
                model.word_language_map[word] = langs
        # print head to confirm load
        print("Loaded model with {} words in word_language_map and {} languages in language_pref_count".format(len(model.word_language_map), len(model.language_pref_count)))

        return model

    def run_pred(self, data):
        preds = []
        for item in tqdm(data):
            output_chars = ""
            char_scores = Counter()
    
            # Convert input data to lowercase if toggle is enabled
            context_words = item.split()
            if self.lowercase:
                context_words = [word.lower() for word in context_words]

            # based on non-last words, get language distribution
            lang_dist = Counter()
            for w in context_words[:-1]:
                if w in self.word_language_map:
                    langs = self.word_language_map[w]
                    lang_dist.update(langs)
            if len(lang_dist) == 0:
                # if no context, just use all languages
                lang_dist.update(self.language_pref_count.keys())

            prefix = context_words[-1]
            if prefix in self.word_language_map:
                # could be a space since a valid word
                # append count of words in each language/total language corpus and normalize across language frequency
                for lang, lang_count in lang_dist.items():
                    prefix_count = self.language_pref_count[lang].get(prefix, 0)
                    char_scores[" "] += prefix_count * lang_count / sum(lang_dist.values())
                
            # based on likelihood of languages, average likelihood of next char across prefixes
            total_lang_count = sum(lang_dist.values())
            for lang, lang_count in lang_dist.items():
                if lang in self.language_pref_count and prefix in self.language_pref_count[lang]:
                    prefix_count = self.language_pref_count[lang][prefix]
                    # Iterate over all words in the language
                    for word, char_count in self.language_pref_count[lang].items():
                        if word.startswith(prefix) and len(word) == len(prefix) + 1:
                            next_char = word[len(prefix)]
                            char_scores[next_char] += prefix_count * char_count * lang_count / total_lang_count
            # choose char with highest scores until output_chars is length 3, we want 3 total predictions for the same next char
            while len(output_chars) < 3:
                if not char_scores:
                    # Handle empty char_scores by appending random character from item
                    # that is not already in output_chars
                    rand_char = random.choice(item)
                    while rand_char in output_chars:
                        rand_char = random.choice(item)
                        # if there are no new characters to choose from, choose random ones
                        if len(set(item) - set(output_chars)) == 0:
                            rand_char = random.choice('abcdefghijklmnopqrstuvwxyz .!?')
                            break
                    output_chars += rand_char
                    LOGGER.warning(f"Empty char_scores for prefix '{item}'. Appending random character '{rand_char}' from input.")
                    continue
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
    parser.add_argument('--languages', nargs='+', default=None,
                        help='ISO 639-1 language codes to train on (default: 40 common languages)')
    parser.add_argument('--max_samples', type=int, default=5000,
                        help='max training samples per language')
    args = parser.parse_args()

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instantiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data(
            languages=args.languages,
            max_samples_per_lang=args.max_samples,
        )
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