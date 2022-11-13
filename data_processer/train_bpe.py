import numpy as np
import jieba
import re
tag = re.compile(u"[^\uac00-\ud7ffa-zA-Z0-9]")
spaces_pun = re.compile(r"[ ]+")


def zh_cut(line):
    lst = " ".join(jieba.lcut(line))
    return lst


def pre_ko(text):
    res = re.sub(tag, lambda x: " " + x.group() + " ", text)
    res = re.sub(spaces_pun, " ", res)
    return res


def ko_cut(line):
    new_line = pre_ko(line)
    new_line = new_line.replace("\n ", "\n")
    return new_line


def split_data(lst, rate=[9.7, 0.2, 0.1]):
    length = len(lst)
    train_ids = int(length * (rate[0]/np.sum(rate)))
    dev_ids = int(length * (np.sum(rate[:2])/np.sum(rate)))
    return lst[:train_ids], lst[train_ids:dev_ids], lst[dev_ids:]


def load(path):
    return [line for line in open(path, "r", encoding="utf-8").readlines()]


for p in ["train.ko.txt", "train.zh.txt"]:
    data = load(p)
    _cut = zh_cut if "zh" in p else ko_cut
    file = p.replace("train", "cut")
    f_out = open(file, "w", encoding="utf-8")
    lines = [_cut(line) for line in data]
    f_out.writelines(lines)
    # for line in data:
    #     lst = _cut(line)
    #     nl = " ".join(lst)
    #     f_out.write(nl)

    f_train = open("data/" + p, "w", encoding="utf-8")
    f_dev = open("data/" + p.replace("train", "dev"), "w", encoding="utf-8")
    f_test = open("data/" + p.replace("train", "test"), "w", encoding="utf-8")

    train, test, dev = split_data(lines)

    f_train.writelines(train)
    f_dev.writelines(dev)
    f_test.writelines(test)