"""Special-token contract shared by ALL tokenizers (character, word, BPE).

The order of SPECIAL_TOKENS defines their IDs, so every tokenizer, every cached token
file and every model built later agrees that 0=pad, 1=unk, 2=bos, 3=eos.
"""
PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
SPECIAL_TOKENS = (PAD, UNK, BOS, EOS)
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
NUM_SPECIAL = len(SPECIAL_TOKENS)
