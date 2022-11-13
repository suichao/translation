import sys
import jieba

# 中文Jieba分词
def jieba_cut(in_file,out_file): 
    out_f = open(out_file,'w',encoding='utf8')
    with open(in_file,'r',encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            cut_line = ' '.join(jieba.cut(line))
            out_f.write(cut_line+'\n')
    out_f.close()


def jieba_cut_reverse(in_file,out_file): 
    out_f = open(out_file,'w',encoding='utf8')
    with open(in_file,'r',encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            cut_line = line.replace(' ','')
            out_f.write(cut_line+'\n')
    out_f.close()

def post_process_seq(seq, bos_idx, eos_idx, output_bos=False, output_eos=False):
    """
    Post-process the decoded sequence.
    """
    eos_pos = len(seq) - 1
    for i, idx in enumerate(seq):
        if idx == eos_idx:
            eos_pos = i
            break
    seq = [
        idx for idx in seq[:eos_pos + 1]
        if (output_bos or idx != bos_idx) and (output_eos or idx != eos_idx)
    ]
    return seq

if __name__ == "__main__":
    in_file = sys.argv[1]
    out_file = sys.argv[2]
    jieba_cut(in_file, out_file)
