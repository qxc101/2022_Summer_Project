import pandas as pd
import glob
import email
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from collections import Counter
from typing import Dict, List, Tuple
import re
from nltk import stem
import os
import codecs
import nltk

stemmer = stem.PorterStemmer()
stopwords = stopwords.words('english')
cut_model = nltk.WordPunctTokenizer()


def listdir(directory: str) -> List:
    """
    A specialized version of os.listdir() that ignores files that
    start with a leading period.

    Especially dismissing .DS_STORE s.
    """
    filelist = os.listdir(directory)
    return [x for x in filelist if not (x.startswith('.'))]


def enron_processor(emails_dir: str, return_list: list) -> list:
    """
    * remove numbers
    * remove stopwords
    * add lables
    """
    dirs = [os.path.join(emails_dir, f) for f in os.listdir(emails_dir)]
    for d in dirs:
        emails = [os.path.join(d, f) for f in os.listdir(d)]
        for mail in emails:
            # print(mail)
            with codecs.open(mail, "rb", encoding='utf_8_sig', errors='ignore') as m:
                email_list = []
                line_str = ""
                for line in m:
                    for word in line:
                        if word.startswith("http"):
                            print(word)
                            word = "URL"
                            print(word)
                        word = stemmer.stem(word)
                    line = re.sub(r'[^a-zA-Z\s]', '', string=line)
                    line = line.lower()
                    line = line.strip()
                    tokens = cut_model.tokenize(line)
                    line = [stemmer.stem(token)
                            for token in tokens if token not in stopwords]

                    line = ' '.join(line)
                    line_str = line_str + line + " "
                email_list.append(line_str)

                if mail.split(".")[-2] == 'spam':
                    email_list.append("spam")
                else:
                    email_list.append("ham")
                email_list.append(mail)
                return_list.append(email_list)


def get_email_content(email_path):
    file = open(email_path, encoding='latin1')
    try:
        msg = email.message_from_file(file)
        for part in msg.walk():
            if part.get_content_type() == 'text/plain':
                return part.get_payload()  # prints the raw text
    except Exception as e:
        print(e)


def get_email_content_bulk(email_paths):
    email_contents = [get_email_content(o) for o in email_paths]
    return email_contents


def remove_null(datas, labels):
    not_null_idx = [i for i, o in enumerate(datas) if o is not None]
    return np.array(datas)[not_null_idx], np.array(labels)[not_null_idx]


def read_text_file(file_path):
    with open(file_path, 'r') as f:
        return f.read().replace("\n", ' ')


def data_extraction(dataset):

    if dataset == 'lingspam':
        df = pd.read_csv('data/lingspam_public/messages.csv')
        x = df.message
        y = df.label
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=99)
        return x_train, x_test, y_train, y_test

    if dataset == 'tutorial':
        path = 'data/spam_data/'
        easy_ham_paths = glob.glob(path + 'easy_ham/*')
        easy_ham_2_paths = glob.glob(path + 'easy_ham_2/*')
        hard_ham_paths = glob.glob(path + 'hard_ham/*')
        spam_paths = glob.glob(path + 'spam/*')
        spam_2_paths = glob.glob(path + 'spam_2/*')
        ham_path = [
            easy_ham_paths,
            easy_ham_2_paths,
            hard_ham_paths
        ]

        spam_path = [
            spam_paths,
            spam_2_paths
        ]
        ham_sample = np.array(
            [train_test_split(o, random_state=999) for o in ham_path])
        ham_train = np.array([])
        ham_test = np.array([])
        for o in ham_sample:
            ham_train = np.concatenate((ham_train, o[0]), axis=0)
            ham_test = np.concatenate((ham_test, o[1]), axis=0)
        spam_sample = np.array(
            [train_test_split(o, random_state=999) for o in spam_path])
        spam_train = np.array([])
        spam_test = np.array([])
        for o in spam_sample:
            spam_train = np.concatenate((spam_train, o[0]), axis=0)
            spam_test = np.concatenate((spam_test, o[1]), axis=0)
        ham_train_label = [0] * ham_train.shape[0]
        spam_train_label = [1] * spam_train.shape[0]
        x_train = np.concatenate((ham_train, spam_train))
        y_train = np.concatenate((ham_train_label, spam_train_label))
        ham_test_label = [0] * ham_test.shape[0]
        spam_test_label = [1] * spam_test.shape[0]
        x_test = np.concatenate((ham_test, spam_test))
        y_test = np.concatenate((ham_test_label, spam_test_label))
        train_shuffle_index = np.random.permutation(
            np.arange(0, x_train.shape[0]))
        test_shuffle_index = np.random.permutation(
            np.arange(0, x_test.shape[0]))
        x_train = x_train[train_shuffle_index]
        y_train = y_train[train_shuffle_index]
        x_test = x_test[test_shuffle_index]
        y_test = y_test[test_shuffle_index]
        x_train = get_email_content_bulk(x_train)
        x_test = get_email_content_bulk(x_test)
        x_train, y_train = remove_null(x_train, y_train)
        x_test, y_test = remove_null(x_test, y_test)
        return x_train, x_test, y_train, y_test

    if dataset == 'enron':

        root_dir = 'data/spampy/datasets/enron'
        emails_dirs = [os.path.join(root_dir, f) for f in listdir(root_dir)]
        return_list = []
        for emails_dir in emails_dirs:
            enron_processor(emails_dir, return_list)

        messages = pd.DataFrame(return_list, columns=[
                                'message', 'label', 'path'])
        messages['label'] = messages['label'].replace('ham', 0)
        messages['label'] = messages['label'].replace('spam', 1)
        messages_label = messages['label']
        x = messages['message']
        y = messages_label
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=99)
        return x_train, x_test, y_train, y_test

    if dataset == 'pu':
        x = []
        y = []
        spam_path = glob.glob("data/pu_corpora_public/*/*/*spmsg*.txt")
        ham_path = glob.glob("data/pu_corpora_public/*/*/*legit*.txt")
        for path in spam_path:
            x.append(read_text_file(path))
            y.append(1)
        for path in ham_path:
            x.append(read_text_file(path))
            y.append(0)
        d = {"X": x, 'Y': y}
        df = pd.DataFrame(data=d)
        df.drop_duplicates(inplace=True)
        x_train, x_test, y_train, y_test = train_test_split(
            df.X, df.Y, test_size=0.2, random_state=99)
        return x_train, x_test, y_train, y_test



