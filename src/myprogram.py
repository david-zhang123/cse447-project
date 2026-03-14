#!/usr/bin/env python
import os
import random
import time
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

CC100_LANGUAGES = [
    "af", "am", "ar", "as", "az", "be", "bg", "bn", "br", "bs",
    "ca", "cs", "cy", "da", "de", "el", "en", "eo", "es", "et",
    "eu", "fa", "ff", "fi", "fr", "fy", "ga", "gd", "gl", "gn",
    "gu", "ha", "he", "hi", "hr", "ht", "hu", "hy", "id", "ig",
    "is", "it", "ja", "jv", "ka", "kk", "km", "kn", "ko", "ku",
    "ky", "la", "lg", "li", "ln", "lo", "lt", "lv", "mg", "mk",
    "ml", "mn", "mr", "ms", "my", "ne", "nl", "no", "ns", "om",
    "or", "pa", "pl", "ps", "pt", "qu", "rm", "ro", "ru", "sa",
    "si", "sk", "sl", "so", "sq", "sr", "ss", "su", "sv", "sw",
    "ta", "te", "th", "tl", "tn", "tr", "ug", "uk", "ur", "uz",
    "vi", "wo", "xh", "yi", "yo", "zh", "zu",
]

DEFAULT_LANGUAGES = [
    "en", "es", "fr", "de", "it", "pt", "nl", "ru", "zh", "ja",
    "ko", "ar", "hi", "bn", "tr", "pl", "vi", "th", "sv", "fi",
    "cs", "ro", "hu", "el", "he", "id", "ms", "uk", "fa", "ta",
    "te", "ml", "ka", "sw", "af", "ur", "sr", "hr", "bg", "sk",
]

class MyModel:
    def __init__(self, vocab_size=None, char_to_idx=None, idx_to_char=None, lowercase=True):
        self.lowercase = lowercase
        self.word_language_map = {}
        self.language_pref_count = {}

    @classmethod
    def load_training_data(cls):
        # load amazon reviews database from huggingface
        # return load_dataset("papluca/language-identification", split="train")

        languages = DEFAULT_LANGUAGES

        data = []
        ds = load_dataset(
            "papluca/language-identification",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        for item in ds:
            text = item["text"].strip()
            data.append({"text": text, "labels": item["labels"]})
        # for lang in tqdm(languages, desc="Loading languages"):
            # try:
                
            #     ds = load_dataset(
            #         "cc100",
            #         lang=lang,
            #         split="train",
            #         streaming=True,
            #         trust_remote_code=True,
            #     )
            #     count = 0
            #     for item in ds:
            #         text = item["text"].strip()
            #         if len(text) < 5:
            #             continue
            #         data.append({"text": text, "labels": lang})
            #         count += 1
            #         if count >= max_samples_per_lang:
            #             break
            #     LOGGER.info(f"Loaded {count} samples for language '{lang}'")
            # except Exception as e:
            #     LOGGER.warning(f"Could not load language '{lang}': {e}")
        random.shuffle(data)
        LOGGER.info(f"Total training samples: {len(data)}")
        return data

    @classmethod
    def load_test_data(cls, fname, lowercase=True):
        test_languages = []  # To store languages for synthetic data
        if fname and fname != 'SYNTHETIC':
            with open(fname) as f:
                test_data = [line.strip() for line in f]
                if lowercase:
                    test_data = [line.lower() for line in test_data]
        else:
            total_data = load_dataset(
                "papluca/language-identification",
                split="test",
                streaming=True,
                trust_remote_code=True,
            )
            # total_data = []
            # for lang in DEFAULT_LANGUAGES:
            #     ds = load_dataset(
            #         "cc100",
            #         lang=lang,
            #         split="train",
            #         streaming=True,
            #         trust_remote_code=True,
            #     )
            #     total_data.extend([{"text": item["text"], "labels": lang} for item in ds])
        
            test_data = [item['text'] for item in total_data]
            test_languages = [item['labels'] for item in total_data]
            correct_next_char = []
            for i in range(len(test_data)):
                test_data[i] = test_data[i].strip()
                if lowercase:
                    test_data[i] = test_data[i].lower()
                if len(test_data[i]) < 2:
                    continue
                index = random.randint(1, len(test_data[i]) - 1)
                correct_next_char.append(test_data[i][index])
                test_data[i] = test_data[i][:index]


            # Write correct next char to file for evaluation
            with open('output/correct_next_char.txt', 'wt') as f:
                for c in correct_next_char:
                    f.write('{}\n'.format(c))

            # Write test languages to file
            with open('output/test_languages.txt', 'wt') as f:
                for lang in test_languages:
                    f.write(f'{lang}\n')

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
        # remove elements that appear less than 5 times from chargram
        for lang, prefix_counts in self.language_pref_count.items():
            for prefix, count in list(prefix_counts.items()):
                if count < 5:
                    del self.language_pref_count[lang][prefix]
                
                
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
        # print head to confirm load
        print("Loaded model with {} words in word_language_map and {} languages in language_pref_count".format(len(model.word_language_map), len(model.language_pref_count)))

        return model

    def run_pred(self, data):
        preds = []
        # with open('output/test_languages.txt') as f:
        #     test_languages = [line.strip() for line in f]

        correct_count = 0  # To calculate language detection accuracy
        for idx, item in enumerate(data): # add tqdm for progress bar
            output_chars = ""

            # Convert input data to lowercase if toggle is enabled
            context_words = item.split()
            if self.lowercase:
                context_words = [word.lower() for word in context_words]

            # based on non-last words, get language distribution
            lang_dist = Counter()
            for w in context_words:
                if w in self.word_language_map:
                    langs = self.word_language_map[w]
                    lang_dist.update(langs)
            if len(lang_dist) == 0:
                # if no context, just use all languages
                lang_dist.update(self.language_pref_count.keys())
            
            # Check if the correct language is in lang_dist
            # correct_language = test_languages[idx]
            # if correct_language in lang_dist:
            #     correct_count += 1

            prefix = context_words[-1] if context_words else ""
            total_lang_count = sum(lang_dist.values())
            char_scores = Counter()

            # add spaces based on likelihood of word being complete, which is based on likelihood of language and prefix being a complete word in that language
            for lang, lang_count in lang_dist.items():
                if prefix in self.language_pref_count[lang]:
                    char_scores[" "] += lang_count * self.language_pref_count[lang][prefix] / total_lang_count
            
            for lang, lang_count in lang_dist.items():
                if prefix in self.language_pref_count[lang]:
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
                    # LOGGER.warning(f"Empty char_scores for prefix '{item}'. Appending random character '{rand_char}' from input.")
                    continue
                next_char = char_scores.most_common(1)[0][0]
                output_chars += next_char
                del char_scores[next_char]
            preds.append(output_chars)

        # Log language detection accuracy
        # language_accuracy = correct_count / len(data)
        # LOGGER.info(f'Language detection accuracy: {language_accuracy:.2%}')
        # with open(os.path.join('output', 'language_accuracy.txt'), 'wt') as f:
        #     f.write(f'Language detection accuracy: {language_accuracy:.2%}\n')

        return preds

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    parser.add_argument('--correct_output', help='path to correct next char file', default='output/correct_next_char.txt')
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
        # print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        
        start_time = time.time()
        # print('Loading model')
        model = MyModel.load(args.work_dir)
        # print('Making predictions')
        pred = model.run_pred(test_data)
        # print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
        elapsed_time = time.time() - start_time

        # Calculate accuracy
        # if os.path.exists(args.correct_output):
        #     with open(args.correct_output) as f:
        #         correct = [line.strip() for line in f]
        #     correct = correct[:len(pred)]  # Ensure lengths match
        #     accuracy = sum(1 for p, c in zip(pred, correct) if p == c) / len(correct)
        #     print(f'Accuracy: {accuracy:.2%}')
        #     LOGGER.info(f'Test accuracy: {accuracy:.2%}')

        #     # Save accuracy to a file
        #     accuracy_file = os.path.join(args.work_dir, 'test_accuracy.txt')
        #     with open(accuracy_file, 'wt') as f:
        #         f.write(f'Accuracy: {accuracy:.2%}\n')

        LOGGER.info(f'Test completed in {elapsed_time:.2f} seconds')
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))