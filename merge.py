import json
    
with open('llama3_tokenizer.json', 'r') as fp:
    llama_tokenizer = json.load(fp)
    
with open('extra_tokenizer.json', 'r') as fp:
    extra_tokenizer = json.load(fp)
    
vocab_merges_map = {}
for merge in extra_tokenizer['model']['merges']:
    merged = ''.join(merge.split(' '))
    if merged in vocab_merges_map:
        vocab_merges_map[merged].append(merge)
    else:
        vocab_merges_map[merged] = [merge]
        
# from https://github.com/openai/gpt-2/blob/master/src/encoder.py#L9
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

tiktoken_map = bytes_to_unicode()
def process_non_en_character(s):
    out = []
    for c in s:
        if ord(c) < 256:
            out.append(tiktoken_map[ord(c)])
        else:
            utf8_encoded = c.encode('utf-8')
            utf8_encoded = [tiktoken_map[byte] for byte in utf8_encoded]
            out.append(''.join(utf8_encoded))
    return ''.join(out)

extend_vocabs = []
extend_merges = []

for v in extra_tokenizer['model']['vocab']:
    if process_non_en_character(v.replace('▁', ' ')) in llama_tokenizer['model']['vocab']:
        continue
    extend_vocabs.append(v)
    if v in vocab_merges_map:
        extend_merges.extend(vocab_merges_map[v])
    if len(extend_vocabs) == 100000:
        break
        
extend_vocabs = [i.replace('▁', ' ') for i in extend_vocabs]
extend_merges = [i for i in extend_merges]

for v in llama_tokenizer['added_tokens']:
    llama_tokenizer['model']['vocab'][v['content']] = v['id']

for i, v in enumerate(extend_vocabs):
    llama_tokenizer['model']['vocab'][process_non_en_character(v)] = 128256 + i
    
for merge in extend_merges:
    first = merge.split(' ')[0].replace('▁', ' ')
    second = merge.split(' ')[1].replace('▁', ' ')
    if first == ' ':
        if second not in llama_tokenizer['model']['vocab']:
            continue
    if process_non_en_character(first) not in llama_tokenizer['model']['vocab'] or process_non_en_character(second) not in llama_tokenizer['model']['vocab']:
        continue
    llama_tokenizer['model']['merges'].append(f"{process_non_en_character(first)} {process_non_en_character(second)}")

with open('merged_tokenizer.json', 'w') as fp:
    json.dump(llama_tokenizer, fp, indent=2, ensure_ascii=False)
    
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('Llama3_merged')

merges = []
for i in range(128256, 256256):
    s = tokenizer.decode(i)
    ss = tokenizer.convert_ids_to_tokens(i)
    tokenized = tokenizer.tokenize(s)
    if len(tokenized) <= 1 or len(tokenized) > 2:
        continue
    assert tokenized[0] in llama_tokenizer['model']['vocab'] and tokenized[1] in llama_tokenizer['model']['vocab']
    if len(tokenized) == 2 and ''.join(tokenized) in llama_tokenizer['model']['vocab']:
        merges.append(' '.join(tokenized))
        
llama_tokenizer['model']['merges'].extend(merges)
with open('merged_tokenizer.json', 'w') as fp:
    json.dump(llama_tokenizer, fp, indent=2, ensure_ascii=False)
