class CharTokenizer:
    def __init__(self, texts=None, specials=["<pad>", "<sos>", "<eos>", "<unk>"]):
        if texts is not None:
            # This path is for initializing during training
            chars = sorted(set("".join(texts)))
            self.itos = specials + chars
            self.stoi = {c: i for i, c in enumerate(self.itos)}
        else:
            # This path is for loading from pickle where `itos` and `stoi` will be overwritten
            self.itos = []
            self.stoi = {}
            
        self.pad_token_id = specials.index("<pad>")
        self.sos_token_id = specials.index("<sos>")
        self.eos_token_id = specials.index("<eos>")
        self.unk_token_id = specials.index("<unk>") if "<unk>" in specials else -1

    def encode(self, text):
        # This is now robust and works for tokenizers with or without an <unk> token.
        unk_id = getattr(self, 'unk_token_id', None)
        tokens = []
        for char in text:
            token = self.stoi.get(char)
            if token is not None:
                tokens.append(token)
            elif unk_id is not None:
                tokens.append(unk_id)
        # If a character is not in the vocab and there's no <unk> token, it is now safely skipped.
        return [self.sos_token_id] + tokens + [self.eos_token_id]

    def decode(self, tokens):
        eos_index = -1
        try:
            eos_index = tokens.index(self.eos_token_id)
        except (ValueError, AttributeError):
            pass # no eos token or not a list
        
        tokens_to_decode = tokens[:eos_index] if eos_index != -1 else tokens
        
        # Ensure itos is populated before decoding
        if not self.itos:
            return ""

        return "".join([self.itos[t] for t in tokens_to_decode if t not in {self.sos_token_id, self.pad_token_id, self.eos_token_id}])

    def __len__(self):
        return len(self.itos) 