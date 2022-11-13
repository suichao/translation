for f in ['trg_vocab.txt', 'src_vocab.txt']:
    vocab = []
    for w in open(f, "r", encoding="utf-8").readlines():
        vocab.append(w.split()[0])
    res = ["<s>", "<e>", "<unk>"] + vocab
    res = [i+"\n" for i in res]
    fp = open(f, 'w', encoding="utf-8")
    fp.writelines(res)