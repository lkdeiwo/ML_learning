import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import spacy
import random
from torch.utils.tensorboard import SummaryWriter
from utils import bleu, save_checkpoint, load_checkpoint

spacy_ger = spacy.load('de_core_news_sm')
spacy_eng = spacy.load('en_core_web_sm')

def tokenizer_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]
def tokenizer_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

german = Field(tokenize=tokenizer_ger, lower=True, init_token='<sos>', eos_token='<eos>')
english = Field(tokenize=tokenizer_eng, lower=True, init_token='<sos>', eos_token='<eos>')
# data
train_data, validation_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(german, english))

german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout_rate):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(dropout_rate)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout_rate)

    def forward(self, x):
        # x.shape: (seq_length, batch_size)

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (seq_len, batch_size, embedding_size)

        outputs, (hidden, cell) = self.rnn(embedding)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, dropout_rate):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(dropout_rate)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        # shape of x: (N) but we want (1, N), 一个单词
        x = x.unsqueeze(0)

        embedding = self.dropout(self.embedding(x))
        # shape of embedding: (1, N, embedding_size)

        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))

        predictions = self.fc(outputs)
        # shape of pred: (1, N, vocal_len)
        predictions = predictions.squeeze(0)

        return predictions, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(english.vocab)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

        hidden, cell = self.encoder(source)

        # grab start token
        x = target[0]
        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[t] = output
            # shape of output = (batch_size, english_vocab_size)
            best_guess = output.argmax(1)
            x = target[t] if random.random() < teacher_force_ratio else best_guess # why

        return outputs

# training
num_epoch = 20
learning_rate = 0.001
batch_size = 64

load_model = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)
output_size = len(english.vocab)
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024
num_layers = 2
enc_dropout = 0.5
dec_dropout = 0.5

# Tensorboard
writer = SummaryWriter(f'runs/loss_plot')
step = 0

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, validation_data, test_data),
    batch_size=batch_size,
    sort_within_batch = True,
    sort_key = lambda x: len(x.src), # 试着去掉这两行，看速度有没有下降
    device=device
)

encoder_net = Encoder(input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout).to(device)
decoder_net = Decoder(input_size_decoder, decoder_embedding_size, hidden_size, output_size, num_layers, dec_dropout).to(device)

model = Seq2Seq(encoder_net, decoder_net).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pad_idx = english.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

for epoch in range(num_epoch):
    print(f'Epoch [{epoch} / {num_epoch}]')

    checkpoint = {}
    save_checkpoint(checkpoint)

    for batch_idx, batch in enumerate(train_iterator):
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)

        output = model(inp_data, target)
        # output shape: (trg_len, batch_size, output_dim)

        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output, target)

        loss.backward()

        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1)
        optimizer.step()

        writer.add_scalar('training loss', loss, global_step=step)
        step += 1







