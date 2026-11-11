"""A tiny test tokenizer. Every byte is a token and decoding is exact, so tests can follow text through the pipeline."""
import re

SPECIAL = ["<|begin_of_text|>", "<|end_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<bos>", "<start_of_turn>", "<end_of_turn>"]
PATTERN = re.compile("(" + "|".join(re.escape(s) for s in SPECIAL) + ")")


class ByteTokenizer:
    vocab_size = 256 + len(SPECIAL)
    eos_token_id = 256 + SPECIAL.index("<|end_of_text|>")

    def convert_tokens_to_ids(self, token):
        return 256 + SPECIAL.index(token)

    def __call__(self, text, add_special_tokens=False):
        ids = []
        for part in PATTERN.split(text):
            if part in SPECIAL:
                ids.append(self.convert_tokens_to_ids(part))
            else:
                ids.extend(part.encode("utf-8"))
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=None):
        if clean_up_tokenization_spaces is not False:
            raise AssertionError("whitespace cleanup must be explicitly off")
        out, buffer = [], bytearray()
        for i in ids:
            if i < 256:
                buffer.append(i)
            else:
                out.append(buffer.decode("utf-8", "replace"))
                buffer = bytearray()
                if not skip_special_tokens:
                    out.append(SPECIAL[i - 256])
        out.append(buffer.decode("utf-8", "replace"))
        return "".join(out)
