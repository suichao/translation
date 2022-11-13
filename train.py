import os
import time
import yaml
import logging
import argparse
import numpy as np
from pprint import pprint
from attrdict import AttrDict
import jieba

import numpy as np
from functools import partial
import paddle
import paddle.distributed as dist
from paddle.io import DataLoader,BatchSampler
from paddlenlp.data import Vocab, Pad
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import TransformerModel, InferTransformerModel, CrossEntropyCriterion, position_encoding_init
from paddlenlp.utils.log import logger

from utils import post_process_seq


# 自定义读取本地数据的方法
def read(src_path, tgt_path, is_predict=False):
    if is_predict:
        with open(src_path, 'r', encoding='utf8') as src_f:
            for src_line in src_f.readlines():
                src_line = src_line.strip()
                if not src_line:
                    continue
                yield {'src':src_line, 'tgt':''}
    else:
        with open(src_path, 'r', encoding='utf8') as src_f, open(tgt_path, 'r', encoding='utf8') as tgt_f:
            for src_line, tgt_line in zip(src_f.readlines(), tgt_f.readlines()):
                src_line = src_line.strip()
                if not src_line:
                    continue
                tgt_line = tgt_line.strip()
                if not tgt_line:
                    continue
                yield {'src':src_line, 'tgt':tgt_line}
 # 过滤掉长度 ≤min_len或者≥max_len 的数据
def min_max_filer(data, max_len, min_len=0):
    # 1 for special tokens.
    data_min_len = min(len(data[0]), len(data[1])) + 1
    data_max_len = max(len(data[0]), len(data[1])) + 1
    return (data_min_len >= min_len) and (data_max_len <= max_len)


# 创建训练集、验证集的dataloader
def create_data_loader(args):
    train_dataset = load_dataset(read, src_path=args.training_file.split(',')[0],
                                 tgt_path=args.training_file.split(',')[1], lazy=False)
    dev_dataset = load_dataset(read, src_path=args.validation_file.split(',')[0],
                               tgt_path=args.validation_file.split(',')[1], lazy=False)

    src_vocab = Vocab.load_vocabulary(
        args.src_vocab_fpath,
        bos_token=args.special_token[0],
        eos_token=args.special_token[1],
        unk_token=args.special_token[2])
    trg_vocab = Vocab.load_vocabulary(
        args.trg_vocab_fpath,
        bos_token=args.special_token[0],
        eos_token=args.special_token[1],
        unk_token=args.special_token[2])

    padding_vocab = (
        lambda x: (x + args.pad_factor - 1) // args.pad_factor * args.pad_factor
    )
    args.src_vocab_size = padding_vocab(len(src_vocab))
    args.trg_vocab_size = padding_vocab(len(trg_vocab))

    def convert_samples(sample):
        source = sample['src'].split()
        target = sample['tgt'].split()

        source = src_vocab.to_indices(source)
        target = trg_vocab.to_indices(target)

        return source, target

    # 训练集dataloader和验证集dataloader
    data_loaders = []
    for i, dataset in enumerate([train_dataset, dev_dataset]):
        dataset = dataset.map(convert_samples, lazy=False).filter(
            partial(min_max_filer, max_len=args.max_length))

        # BatchSampler: https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/BatchSampler_cn.html
        batch_sampler = BatchSampler(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

        # DataLoader: https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/DataLoader_cn.html
        data_loader = DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            collate_fn=partial(
                prepare_train_input,
                bos_idx=args.bos_idx,
                eos_idx=args.eos_idx,
                pad_idx=args.bos_idx),
            num_workers=0,
            return_list=True)
        data_loaders.append(data_loader)

    return data_loaders


def prepare_train_input(insts, bos_idx, eos_idx, pad_idx):
    """
    Put all padded data needed by training into a list.
    """
    word_pad = Pad(pad_idx)
    src_word = word_pad([inst[0] + [eos_idx] for inst in insts])
    trg_word = word_pad([[bos_idx] + inst[1] for inst in insts])
    lbl_word = np.expand_dims(
        word_pad([inst[1] + [eos_idx] for inst in insts]), axis=2)

    data_inputs = [src_word, trg_word, lbl_word]

    return data_inputs


# 创建测试集的dataloader，原理步骤同上（创建训练集、验证集的dataloader）
def create_infer_loader(args):
    dataset = load_dataset(read, src_path=args.predict_file, tgt_path=None, is_predict=True, lazy=False)

    src_vocab = Vocab.load_vocabulary(
        args.src_vocab_fpath,
        bos_token=args.special_token[0],
        eos_token=args.special_token[1],
        unk_token=args.special_token[2])
    trg_vocab = Vocab.load_vocabulary(
        args.trg_vocab_fpath,
        bos_token=args.special_token[0],
        eos_token=args.special_token[1],
        unk_token=args.special_token[2])

    padding_vocab = (
        lambda x: (x + args.pad_factor - 1) // args.pad_factor * args.pad_factor
    )
    args.src_vocab_size = padding_vocab(len(src_vocab))
    args.trg_vocab_size = padding_vocab(len(trg_vocab))

    def convert_samples(sample):
        source = sample['src'].split()
        target = sample['tgt'].split()

        source = src_vocab.to_indices(source)
        target = trg_vocab.to_indices(target)

        return source, target

    dataset = dataset.map(convert_samples, lazy=False)

    # BatchSampler: https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/BatchSampler_cn.html
    batch_sampler = BatchSampler(dataset, batch_size=args.infer_batch_size, drop_last=False)

    # DataLoader: https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/DataLoader_cn.html
    data_loader = DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=partial(
            prepare_infer_input,
            bos_idx=args.bos_idx,
            eos_idx=args.eos_idx,
            pad_idx=args.bos_idx),
        num_workers=0,
        return_list=True)
    return data_loader, trg_vocab.to_tokens


def prepare_infer_input(insts, bos_idx, eos_idx, pad_idx):
    """
    Put all padded data needed by beam search decoder into a list.
    """
    word_pad = Pad(pad_idx)
    src_word = word_pad([inst[0] + [eos_idx] for inst in insts])

    return [src_word, ]


def do_train(args):
    if args.use_gpu:
        place = "gpu"
    else:
        place = "cpu"
    paddle.set_device(place)
    # Set seed for CE
    random_seed = eval(str(args.random_seed))
    if random_seed is not None:
        paddle.seed(random_seed)

    # Define data loader
    (train_loader), (eval_loader) = create_data_loader(args)

    # Define model
    transformer = TransformerModel(
        src_vocab_size=args.src_vocab_size,
        trg_vocab_size=args.trg_vocab_size,
        max_length=args.max_length + 1,
        n_layer=args.n_layer,
        # num_encoder_layers=args.n_layer,
        # num_decoder_layers=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        d_inner_hid=args.d_inner_hid,
        dropout=args.dropout,
        weight_sharing=args.weight_sharing,
        bos_id=args.bos_idx,
        eos_id=args.eos_idx)

    # Define loss
    criterion = CrossEntropyCriterion(args.label_smooth_eps, args.bos_idx)

    scheduler = paddle.optimizer.lr.NoamDecay(
        args.d_model, args.warmup_steps, args.learning_rate, last_epoch=0)

    # Define optimizer
    optimizer = paddle.optimizer.Adam(
        learning_rate=scheduler,
        beta1=args.beta1,
        beta2=args.beta2,
        epsilon=float(args.eps),
        parameters=transformer.parameters())

    step_idx = 0

    # Train loop
    for pass_id in range(args.epoch):
        batch_id = 0
        for input_data in train_loader:

            (src_word, trg_word, lbl_word) = input_data

            logits = transformer(src_word=src_word, trg_word=trg_word)

            sum_cost, avg_cost, token_num = criterion(logits, lbl_word)

            # 计算梯度
            avg_cost.backward()
            # 更新参数
            optimizer.step()
            # 梯度清零
            optimizer.clear_grad()

            if (step_idx + 1) % args.print_step == 0 or step_idx == 0:
                total_avg_cost = avg_cost.numpy()
                logger.info(
                    "step_idx: %d, epoch: %d, batch: %d, avg loss: %f, "
                    " ppl: %f " %
                    (step_idx, pass_id, batch_id, total_avg_cost,
                     np.exp([min(total_avg_cost, 100)])))

            if (step_idx + 1) % args.save_step == 0:
                # Validation
                transformer.eval()
                total_sum_cost = 0
                total_token_num = 0
                with paddle.no_grad():
                    for input_data in eval_loader:
                        (src_word, trg_word, lbl_word) = input_data
                        logits = transformer(
                            src_word=src_word, trg_word=trg_word)
                        sum_cost, avg_cost, token_num = criterion(logits,
                                                                  lbl_word)
                        total_sum_cost += sum_cost.numpy()
                        total_token_num += token_num.numpy()
                        total_avg_cost = total_sum_cost / total_token_num
                    logger.info("validation, step_idx: %d, avg loss: %f, "
                                " ppl: %f" %
                                (step_idx, total_avg_cost,
                                 np.exp([min(total_avg_cost, 100)])))
                transformer.train()

                if args.save_model:
                    model_dir = os.path.join(args.save_model,
                                             "step_" + str(step_idx))
                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)
                    paddle.save(transformer.state_dict(),
                                os.path.join(model_dir, "transformer.pdparams"))
                    paddle.save(optimizer.state_dict(),
                                os.path.join(model_dir, "transformer.pdopt"))
            batch_id += 1
            step_idx += 1
            scheduler.step()

    if args.save_model:
        model_dir = os.path.join(args.save_model, "step_final")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        paddle.save(transformer.state_dict(),
                    os.path.join(model_dir, "transformer.pdparams"))
        paddle.save(optimizer.state_dict(),
                    os.path.join(model_dir, "transformer.pdopt"))


# 创建测试集的dataloader，原理步骤同上（创建训练集、验证集的dataloader）
def create_infer_loader(args):
    dataset = load_dataset(read, src_path=args.predict_file, tgt_path=None, is_predict=True, lazy=False)

    src_vocab = Vocab.load_vocabulary(
        args.src_vocab_fpath,
        bos_token=args.special_token[0],
        eos_token=args.special_token[1],
        unk_token=args.special_token[2])
    trg_vocab = Vocab.load_vocabulary(
        args.trg_vocab_fpath,
        bos_token=args.special_token[0],
        eos_token=args.special_token[1],
        unk_token=args.special_token[2])

    padding_vocab = (
        lambda x: (x + args.pad_factor - 1) // args.pad_factor * args.pad_factor
    )
    args.src_vocab_size = padding_vocab(len(src_vocab))
    args.trg_vocab_size = padding_vocab(len(trg_vocab))

    def convert_samples(sample):
        source = sample['src'].split()
        target = sample['tgt'].split()

        source = src_vocab.to_indices(source)
        target = trg_vocab.to_indices(target)

        return source, target

    dataset = dataset.map(convert_samples, lazy=False)

    # BatchSampler: https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/BatchSampler_cn.html
    batch_sampler = BatchSampler(dataset, batch_size=args.infer_batch_size, drop_last=False)

    # DataLoader: https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/DataLoader_cn.html
    data_loader = DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=partial(
            prepare_infer_input,
            bos_idx=args.bos_idx,
            eos_idx=args.eos_idx,
            pad_idx=args.bos_idx),
        num_workers=0,
        return_list=True)
    return data_loader, trg_vocab.to_tokens


def prepare_infer_input(insts, bos_idx, eos_idx, pad_idx):
    """
    Put all padded data needed by beam search decoder into a list.
    """
    word_pad = Pad(pad_idx)
    src_word = word_pad([inst[0] + [eos_idx] for inst in insts])

    return [src_word, ]


def do_predict(args):
    if args.use_gpu:
        place = "gpu"
    else:
        place = "cpu"
    paddle.set_device(place)

    # Define data loader
    test_loader, to_tokens = create_infer_loader(args)

    # Define model
    transformer = InferTransformerModel(
        src_vocab_size=args.src_vocab_size,
        trg_vocab_size=args.trg_vocab_size,
        max_length=args.max_length + 1,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        d_inner_hid=args.d_inner_hid,
        dropout=args.dropout,
        weight_sharing=args.weight_sharing,
        bos_id=args.bos_idx,
        eos_id=args.eos_idx,
        beam_size=args.beam_size,
        max_out_len=args.max_out_len)

    # Load the trained model
    assert args.init_from_params, (
        "Please set init_from_params to load the infer model.")

    model_dict = paddle.load(
        os.path.join(args.init_from_params, "transformer.pdparams"))

    # To avoid a longer length than training, reset the size of position
    # encoding to max_length
    model_dict["encoder.pos_encoder.weight"] = position_encoding_init(
        args.max_length + 1, args.d_model)
    model_dict["decoder.pos_encoder.weight"] = position_encoding_init(
        args.max_length + 1, args.d_model)
    transformer.load_dict(model_dict)

    # Set evaluate mode
    transformer.eval()

    f = open(args.output_file, "w", encoding="utf-8")
    with paddle.no_grad():
        for (src_word, ) in test_loader:
            finished_seq = transformer(src_word=src_word)
            finished_seq = finished_seq.numpy().transpose([0, 2, 1])
            for ins in finished_seq:
                for beam_idx, beam in enumerate(ins):
                    if beam_idx >= args.n_best:
                        break
                    id_list = post_process_seq(beam, args.bos_idx, args.eos_idx)
                    word_list = to_tokens(id_list)
                    sequence = " ".join(word_list) + "\n"
                    f.write(sequence)
    f.close()

if __name__ == '__main__':
    yaml_file = 'transformer.base.yaml'
    with open(yaml_file, 'rt') as f:
        args = AttrDict(yaml.safe_load(f))
        pprint(args)
        # do_train(args)
        do_predict(args)