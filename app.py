import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pydub import AudioSegment
from pydub.playback import play
from io import BytesIO
import base64
import os
from gtts import gTTS # Google Text-to-Speech
from googletrans import Translator # Import Translator
import math
import gdown

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize Google Translator
translator = Translator()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch is using device: {device}")

### --- Reusable Components ---

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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

# --- Translation Model ---
class TranslationTransformer(nn.Module):
    # Renamed to avoid confusion with the chatbot model
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_layers,
            num_decoder_layers=num_layers, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
    
    def forward(self, src, tgt):
        # This forward is for training, inference uses a different loop
        pass

# --- Chatbot Model ---
class ChatbotTransformer(nn.Module):
    # This class now matches your training script
    def __init__(self, input_vocab, output_vocab, d_model=512, nhead=8, num_layers=6, q_pad_id=0, a_pad_id=0):
        super().__init__()
        self.q_pad_id = q_pad_id
        self.a_pad_id = a_pad_id
        self.encoder_embed = nn.Embedding(input_vocab, d_model)
        self.decoder_embed = nn.Embedding(output_vocab, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_layers,
            num_decoder_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(d_model, output_vocab)
# Ensure folders exist
os.makedirs('Chatbot', exist_ok=True)
os.makedirs('Translation', exist_ok=True)

def download_model(file_id, output_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading {output_path} from Google Drive...")
    try:
        gdown.download(url, output_path, quiet=False)
    except Exception as e:
        print(f"Failed to download {output_path}: {e}")



# Download all required files before loading models/tokenizers
download_model('1KK-Lbu5UrVLwq4LX9Qy3doR_CwdSP0ht', 'Chatbot/chatbot_best_model.pt')
download_model('1LwhKdyYSoiPJmR6OzYGkGRFHbiqhHJmK', 'Chatbot/chatbot_transformer.pth')
download_model('148zyLpKhx_MYRzVsQEilYnHapjorq9QG', 'Chatbot/q_tokenizer.pkl')
download_model('1x8rPklKFVdB9bcDmTkpPIeZagk0F5FCq', 'Chatbot/a_tokenizer.pkl')
download_model('18uUpGYshH-6IFgE6GDUgpgSl0rldUi4L', 'Translation/src_tokenizer.pkl')
download_model('184xm1fCkSNJBAQUE7sqeKoWpWosHArzF', 'Translation/tgt_tokenizer.pkl')
download_model('1YrnMu7GRcq5lYv5KVcGzjTWWO_sTtxQo', 'Translation/baybayin_transformer_model.pth')

# --- Loading Models and Tokenizers ---
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load Translation Model
try:
    src_tokenizer_path = os.path.join(current_dir, "Translation", "src_tokenizer.pkl")
    tgt_tokenizer_path = os.path.join(current_dir, "Translation", "tgt_tokenizer.pkl")
    translation_model_path = os.path.join(current_dir, "Translation", "baybayin_transformer_model.pth")

    with open(src_tokenizer_path, "rb") as f:
        translation_src_tokenizer = pickle.load(f)
    with open(tgt_tokenizer_path, "rb") as f:
        translation_tgt_tokenizer = pickle.load(f)

    # Note: Ensure these parameters match how the translation model was trained
    translation_model = TranslationTransformer(len(translation_src_tokenizer), len(translation_tgt_tokenizer)).to(device)
    translation_model.load_state_dict(torch.load(translation_model_path, map_location=device))
    translation_model.eval()
    print("Translation model and tokenizers loaded successfully!")
    print(f"Translation model loaded on device: {next(translation_model.parameters()).device}")
except Exception as e:
    print(f"Error loading translation model or tokenizers: {e}")
    translation_model = None

# Load Chatbot Model
try:
    q_tokenizer_path = os.path.join(current_dir, "Chatbot", "q_tokenizer.pkl")
    a_tokenizer_path = os.path.join(current_dir, "Chatbot", "a_tokenizer.pkl")
    chatbot_model_path = os.path.join(current_dir, "Chatbot", "chatbot_best_model.pt") # Using the best model

    with open(q_tokenizer_path, "rb") as f:
        chatbot_q_tokenizer = pickle.load(f)
    with open(a_tokenizer_path, "rb") as f:
        chatbot_a_tokenizer = pickle.load(f)

    # Parameters MUST match the successful training script
    d_model = 512
    nhead = 8
    num_layers = 6

    chatbot_model = ChatbotTransformer(
        input_vocab=len(chatbot_q_tokenizer),
        output_vocab=len(chatbot_a_tokenizer),
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        q_pad_id=chatbot_q_tokenizer.pad_token_id,
        a_pad_id=chatbot_a_tokenizer.pad_token_id
    ).to(device)

    chatbot_model.load_state_dict(torch.load(chatbot_model_path, map_location=device))
    chatbot_model.eval()
    print("Chatbot model and tokenizers loaded successfully!")
    print(f"Chatbot model loaded on device: {next(chatbot_model.parameters()).device}")
except Exception as e:
    print(f"Error loading chatbot model or tokenizers: {e}")
    chatbot_model = None


### --- Inference Functions ---

def translate_beam_search(text, model, src_tokenizer, tgt_tokenizer, max_len=100, beam_width=5):
    # This is the full implementation of the beam search for translation.
    model.eval()
    src = torch.tensor(src_tokenizer.encode(text)).unsqueeze(0).to(device)
    src_pad_mask = src == src_tokenizer.pad_token_id
    memory = model.transformer.encoder(model.pos_encoding(model.src_embedding(src)), src_key_padding_mask=src_pad_mask)

    sequences = [[ [tgt_tokenizer.sos_token_id], 0.0 ]]
    completed_sequences = []

    for _ in range(max_len):
        all_candidates = []
        for seq, score in sequences:
            if seq[-1] == tgt_tokenizer.eos_token_id:
                completed_sequences.append((seq, score))
                continue

            tgt = torch.tensor([seq]).to(device)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(len(seq)).to(device)
            out = model.transformer.decoder(model.pos_encoding(model.tgt_embedding(tgt)), memory, tgt_mask=tgt_mask)
            out = model.fc_out(out[:, -1])
            log_probs = F.log_softmax(out, dim=-1)

            topk_log_probs, topk_indices = torch.topk(log_probs, beam_width, dim=-1)
            for i in range(beam_width):
                next_token = topk_indices[0][i].item()
                next_score = score + topk_log_probs[0][i].item()
                all_candidates.append((seq + [next_token], next_score))

        if not all_candidates:
            break

        sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]

        # Move completed sequences
        newly_completed = [s for s in sequences if s[0][-1] == tgt_tokenizer.eos_token_id]
        completed_sequences.extend(newly_completed)
        sequences = [s for s in sequences if s[0][-1] != tgt_tokenizer.eos_token_id]

        if not sequences:
            break

    if not completed_sequences:
        completed_sequences.extend(sequences)
    if not completed_sequences:
        return "Sorry, I could not generate a translation."

    best_sequence = sorted(completed_sequences, key=lambda x: x[1] / len(x[0]), reverse=True)[0][0]
    return tgt_tokenizer.decode(best_sequence)

def chatbot_response(question, model, q_tokenizer, a_tokenizer, beam_width=5, max_len=150):
    # This function now matches your successful training script
    model.eval()
    src = torch.tensor(q_tokenizer.encode(question)).unsqueeze(0).to(device)
    src_pad_mask = src == q_tokenizer.pad_token_id
    memory = model.transformer.encoder(model.pos_encoding(model.encoder_embed(src)), src_key_padding_mask=src_pad_mask)

    sequences = [[ [a_tokenizer.sos_token_id], 0.0 ]]
    completed_sequences = []

    for _ in range(max_len):
        all_candidates = []
        for seq, score in sequences:
            if seq[-1] == a_tokenizer.eos_token_id:
                completed_sequences.append((seq, score))
                continue

            tgt = torch.tensor([seq]).to(device)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(len(seq)).to(device)
            out = model.transformer.decoder(model.pos_encoding(model.decoder_embed(tgt)), memory, tgt_mask=tgt_mask)
            out = model.fc_out(out[:, -1])
            log_probs = F.log_softmax(out, dim=-1)

            topk_log_probs, topk_indices = torch.topk(log_probs, beam_width, dim=-1)
            for i in range(beam_width):
                next_token = topk_indices[0][i].item()
                next_score = score + topk_log_probs[0][i].item()
                all_candidates.append((seq + [next_token], next_score))

        if not all_candidates:
            break
        
        sequences = sorted(all_candidates, key=lambda x: x[1] / len(x[0]), reverse=True)[:beam_width]
        live_sequences = [s for s in sequences if s[0][-1] != a_tokenizer.eos_token_id]
        
        # Add completed sequences from this step
        completed_sequences.extend([s for s in sequences if s[0][-1] == a_tokenizer.eos_token_id])

        sequences = live_sequences
        if not sequences:
            break

    if not completed_sequences:
         completed_sequences.extend(sequences)
    if not completed_sequences:
        return "Sorry, I could not generate a response."

    best_sequence = sorted(completed_sequences, key=lambda x: x[1] / len(x[0]), reverse=True)[0][0]
    return a_tokenizer.decode(best_sequence)


### --- API Endpoints ---

@app.route('/translate', methods=['POST'])
def translate_text():
    if not translation_model or not translation_src_tokenizer or not translation_tgt_tokenizer:
        return jsonify({"error": "Model or tokenizers not loaded."}), 500

    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({"error": "No text provided"}), 400

    debug_messages = []
    debug_messages.append(f"Received text for translation: '{text}'")

    try:
        # Detect language and translate to Filipino if English
        detected_lang = translator.detect(text).lang
        debug_messages.append(f"Detected language: {detected_lang}")

        filipino_text = text
        if detected_lang == 'en':
            translation = translator.translate(text, dest='tl')
            filipino_text = translation.text
            debug_messages.append(f"Translated English to Filipino: '{filipino_text}'")

        # Correctly call the translation function with the Filipino text
        translated_baybayin_text = translate_beam_search(filipino_text, translation_model, translation_src_tokenizer, translation_tgt_tokenizer)
        debug_messages.append(f"Translated Baybayin: '{translated_baybayin_text}'")

        # Generate speech audio
        tts = gTTS(text=filipino_text, lang='tl')  # Use the Filipino word and Filipino language code
        audio_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_output")
        audio_file_path = os.path.join(audio_output_dir, "translation_audio.mp3")
        os.makedirs(os.path.dirname(audio_file_path), exist_ok=True)
        tts.save(audio_file_path)
        debug_messages.append(f"Generated audio file: '{audio_file_path}'")

        audio_url_for_frontend = f"http://127.0.0.1:5050/audio/{os.path.basename(audio_file_path)}"
        
        return jsonify({
            "baybayin": translated_baybayin_text,
            "filipino": filipino_text,
            "audio_url": audio_url_for_frontend,
            "debug_messages": debug_messages
        })
    except Exception as e:
        import traceback # Import traceback for detailed error logging
        error_traceback = traceback.format_exc()
        debug_messages.append(f"Error during translation or audio generation: {e}\nTraceback:\n{error_traceback}")
        return jsonify({
            "error": "An error occurred during translation.",
            "details": str(e),
            "debug_messages": debug_messages
        }), 500

@app.route('/audio/<filename>')
def serve_audio(filename):
    audio_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_output")
    return send_from_directory(audio_output_dir, filename)

@app.route('/chatbot', methods=['POST'])
def handle_chatbot():
    if not chatbot_model:
        return jsonify({"error": "Chatbot model is not loaded."}), 500
        
    data = request.get_json()
    question = data.get('question', '')
    if not question:
        return jsonify({"error": "No question provided"}), 400
    try:
        answer = chatbot_response(question, chatbot_model, chatbot_q_tokenizer, chatbot_a_tokenizer)
        return jsonify({"answer": answer})
    except Exception as e:
        print(f"Chatbot error: {e}")
        return jsonify({"error": "An error occurred in the chatbot."}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    app.run(host='0.0.0.0', debug=False, port=port) 