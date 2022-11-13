#!/bin/bash
# 参考 https://github.com/rsennrich/subword-nmt

# 解压数据
if [ ! -d train_dev_test ]; then
    echo "Decompress train_dev_test data..."
    tar -zxf train_dev_test.tar.gz
fi

folder="train_dev_test"
echo "jieba tokenize..."
python utils.py ${folder}/casia2015_ch_train.txt ${folder}/casia2015_ch_train.cut.txt


echo "source learn-bpe and apply-bpe..."
subword-nmt learn-bpe -s 32000 < ${folder}/casia2015_ch_train.cut.txt > ${folder}/bpe.ch.32000
subword-nmt apply-bpe -c ${folder}/bpe.ch.32000 < ${folder}/casia2015_ch_train.cut.txt > ${folder}/train.ch.bpe
subword-nmt apply-bpe -c ${folder}/bpe.ch.32000 < ${folder}/casia2015_ch_dev.txt > ${folder}/dev.ch.bpe

echo "target learn-bpe and apply-bpe..."
subword-nmt learn-bpe -s 32000 < ${folder}/casia2015_en_train.txt > ${folder}/bpe.en.32000
subword-nmt apply-bpe -c ${folder}/bpe.en.32000 < ${folder}/casia2015_en_train.txt > ${folder}/train.en.bpe
subword-nmt apply-bpe -c ${folder}/bpe.en.32000 < ${folder}/casia2015_en_dev.txt > ${folder}/dev.en.bpe

echo "source get-vocab. if loading pretrained model, use its vocab."
subword-nmt  get-vocab -i ${folder}/train.ch.bpe -o ${folder}/temp
echo -e "<s>\n<e>\n<unk>" > ${folder}/vocab.ch.src
cat ${folder}/temp | cut -f1 -d ' ' >> ${folder}/vocab.ch.src
rm -f ${folder}/temp

echo "target get-vocab. if loading pretrained model, use its vocab."
subword-nmt  get-vocab -i ${folder}/train.en.bpe -o ${folder}/temp
echo -e "<s>\n<e>\n<unk>" > ${folder}/vocab.en.tgt
cat ${folder}/temp | cut -f1 -d ' ' >> ${folder}/vocab.en.tgt
rm -f ${folder}/temp

echo "Over."
