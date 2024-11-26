import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import pandas as pd
import os
import random
import csv
import nltk
import sys


dataset_path = sys.argv[1]


def init_weights(m):
    if type(m) in [nn.Conv2d, nn.Linear]:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class Vocabulary:
    def __init__(self):
        self.char2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2char = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.idx = 4  # Start indexing from 4

    def add_sentence(self, sentence):
        for char in sentence:
            self.add_char(char)

    def add_char(self, char):
        if char not in self.char2idx:
            self.char2idx[char] = self.idx
            self.idx2char[self.idx] = char
            self.idx += 1

    def __len__(self):
        return len(self.char2idx)

    def string_to_indices(self, sentence, add_eos=False):
        indices = [self.char2idx.get(char, self.char2idx["<UNK>"]) for char in sentence]
        if add_eos:
            indices.append(self.char2idx["<EOS>"])
        return indices

    def indices_to_string(self, indices):
        return ''.join(self.idx2char.get(idx, "<UNK>") for idx in indices if idx not in (self.char2idx["<PAD>"], self.char2idx["<SOS>"], self.char2idx["<EOS>"]))


class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=5)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=5)
        self.pool5 = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(3)
        self.apply(init_weights)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = F.relu(self.conv5(x))
        x = self.pool5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=512, hidden_dim=512, context_size=512):
        super(LSTMDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim + context_size, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.init_h = nn.Linear(context_size, hidden_dim)  # Transform context vector
        self.apply(init_weights)

    def forward(self, input_token, hidden_state=None, context_vector=None):
        if hidden_state is None:
            h0 = self.init_h(context_vector).unsqueeze(0)
            c0 = torch.zeros_like(h0)
            hidden_state = (h0, c0)

        embeddings = self.embedding(input_token)
        # Concatenate embeddings with context vector
        context_vector = context_vector.unsqueeze(1).repeat(1, embeddings.size(1), 1)
        lstm_input = torch.cat((embeddings, context_vector), dim=2)
        
        lstm_out, hidden_state = self.lstm(lstm_input, hidden_state)
        outputs = self.fc(lstm_out)
        return outputs, hidden_state



class MathExpressionsDataset(Dataset):
    def __init__(self, csv_file, images_folder, vocab, transform, expect_zero = 0):
        self.data_frame = pd.read_csv(csv_file, skiprows=1, names=['image_name', 'latex_code'])
        self.images_folder = images_folder
        self.transform = transform
        self.vocab = vocab
        if expect_zero:
            replace_string = '[x]'
            for index, row in self.data_frame.iterrows():
                self.data_frame.at[index, 'latex_code'] = replace_string  
        self.max_length = max(len(latex_code) for latex_code in self.data_frame['latex_code']) + 2  # +2 for <SOS> and <EOS>

    def __len__(self):
        return len(self.data_frame)
    
    def preprocess_images(self):
        for idx, image_name in enumerate(self.data_frame['image_name']):
            image_path = os.path.join(self.images_folder, image_name)
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            self.images[idx] = image

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_folder, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        # image = self.images[idx]
        latex_code = self.data_frame.iloc[idx, 1]
        indices = [self.vocab.char2idx['<SOS>']] + self.vocab.string_to_indices(latex_code) + [self.vocab.char2idx['<EOS>']]
        padded_indices = np.pad(indices, (0, self.max_length - len(indices)), 'constant', constant_values=self.vocab.char2idx['<PAD>'])
        image = self.transform(image)
        return image, torch.tensor(padded_indices, dtype=torch.long)
    

def sbleu(GT,PRED):
    score = 0
    for i in range(len(GT)):
        Lgt = len(GT[i].split(' '))
        if Lgt > 4 :
            cscore = nltk.translate.bleu_score.sentence_bleu([GT[i].split(' ')],PRED[i].split(' '),weights=(0.25,0.25,0.25,0.25),smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method4)
        else:
            weight_lst = tuple([1.0/Lgt]*Lgt)
            cscore = nltk.translate.bleu_score.sentence_bleu([GT[i].split(' ')],PRED[i].split(' '),weights=weight_lst,smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method4)
        score += cscore
    return score/(len(GT))

def validate_and_calculate_bleu(loader, encoder, decoder, vocab, device, cal_bleu = True):
    encoder.eval()
    decoder.eval()

    references = []
    hypotheses = []
    predictions = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            context_vectors = encoder(images)
            hidden = None

            # Start with <SOS> token
            decoder_input = torch.full((images.size(0), 1), vocab.char2idx['<SOS>'], dtype=torch.long, device=device)
            predicted_indices = []

            eos_reached = torch.zeros(images.size(0), dtype=torch.bool, device=device)

            for t in range(100):  # Assuming max length of prediction is 100
                decoder_output, hidden = decoder(decoder_input, hidden, context_vectors)
                top1 = decoder_output.argmax(2).squeeze(1)
                eos_reached |= (top1 == vocab.char2idx['<EOS>'])
                decoder_input = top1.unsqueeze(1)
                
                predicted_indices.append(top1.cpu().numpy())

                if eos_reached.all():
                    break

            predicted_indices = np.array(predicted_indices).T  # Transpose to get batch-wise indexing
            for idx_seq in predicted_indices:
                hypotheses.append(vocab.indices_to_string(idx_seq))

            # Convert target indices to LaTeX strings (ground truth)
            for target_seq in targets.cpu().numpy():
                references.append(vocab.indices_to_string(target_seq))

    bleu_score = 0
    # Calculate BLEU score
    if cal_bleu:
        bleu_score = sbleu(references, hypotheses), predictions
    return bleu_score, hypotheses




def train(loader, encoder, decoder, criterion, optimizer, num_epochs, vocab, device, clip_value= 0):
    encoder.train()
    decoder.train()
    encoder.to(device)
    decoder.to(device)

    for epoch in range(num_epochs):
        total_loss = 0
        for images, targets in loader:
            if torch.isnan(images).any() or torch.isnan(targets).any():
                print("NaN detected in input data")
                continue

            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()

            context_vectors = encoder(images)
            input_sequences = targets[:, :-1]  # Exclude the <EOS> token
            target_sequences = targets[:, 1:]  # Exclude the <SOS> token

            decoder_input = torch.full((targets.size(0), 1), vocab.char2idx['<SOS>'], dtype=torch.long, device=device)
            loss = 0
            sequence_length = input_sequences.size(1)

            hidden = None
            actual_lengths = (targets != vocab.char2idx['<PAD>']).sum(dim=1)
            max_length = actual_lengths.max()
            for t in range(max_length-1):
                decoder_output, hidden = decoder(decoder_input, hidden, context_vectors)
                # Check for NaNs in model output
                if torch.isnan(decoder_output).any():
                    print("NaN detected in model output")
                    print("Decoder output:", decoder_output)
                    break  # Optionally, break the loop to avoid further NaN propagation

                loss_t = criterion(decoder_output.squeeze(1), target_sequences[:, t])
                loss += loss_t
                teacher_force = random.random() < 0.5
                top1 = decoder_output.argmax(2).squeeze(1)
                decoder_input = (target_sequences[:, t] if teacher_force else top1).unsqueeze(1)

            non_pad_mask = target_sequences != vocab.char2idx['<PAD>']
            non_pad_elements = non_pad_mask.sum()
            loss /= non_pad_elements
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
#             print(f'Epoch [{epoch+1}/{num_epochs}], Batch Loss: {loss.item():.4f}')

        print(f'Epoch [{epoch+1}/{num_epochs}], Total Loss: {total_loss:.4f}')

    return encoder, decoder




image_size = 224  # Resize image to 224x224

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

csv_path_train = "{}/SyntheticData/train.csv".format(dataset_path)
csv_path_val = "{}/SyntheticData/val.csv".format(dataset_path)
csv_path_test = "{}/SyntheticData/test.csv".format(dataset_path)
image_folder = "{}/SyntheticData/images".format(dataset_path)

# Assuming the new dataset paths
csv_path_handwritten_train = "{}/HandwrittenData/train_hw.csv".format(dataset_path)
csv_path_handwritten_val = "{}/HandwrittenData/val_hw.csv".format(dataset_path)
handwritten_image_folder = "{}/HandwrittenData/images/train".format(dataset_path)

sample_path = "{}/sample_sub.csv".format(dataset_path)
sample_image = "{}/HandwrittenData/images/test".format(dataset_path)


# Vocabulary
temp_dataset = pd.read_csv(csv_path_train, names=['image_name', 'latex_code'])
vocab = Vocabulary()
for latex_code in temp_dataset['latex_code']:
    vocab.add_sentence(latex_code)

# Datasets and DataLoaders
train_dataset = MathExpressionsDataset(csv_file=csv_path_train, images_folder=image_folder, vocab=vocab, transform=transform)
# val_dataset = MathExpressionsDataset(csv_file=csv_path_val, images_folder=image_folder, vocab=vocab, transform=transform)
test_dataset = MathExpressionsDataset(csv_file=csv_path_test, images_folder=image_folder, vocab=vocab, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)


# Model, Loss, and Optimizer
encoder = CNNEncoder()
decoder = LSTMDecoder(len(vocab))

optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=vocab.char2idx['<PAD>'])

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 10
# Assuming you have a 'val_loader'
encoder, decoder = train(train_loader, encoder, decoder, criterion, optimizer, num_epochs, vocab, device)


bleu_score, predictions = validate_and_calculate_bleu(test_loader, encoder, decoder, vocab, device)
print(f'BLEU Score on Test Set: {bleu_score}')

dataset = pd.read_csv(csv_path_test)
image_names = dataset['image'].tolist() 

output_path = "pred1b.csv"
with open(output_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['image_id', 'formula'])  
    for img_name, prediction in zip(image_names, predictions):
        writer.writerow([img_name, prediction])

print(f"Predictions written to {output_path}")



sample_dataset = MathExpressionsDataset(csv_file = sample_path, images_folder = sample_image , vocab=vocab, transform=transform, expect_zero=1)
sample_loader = DataLoader(sample_dataset, batch_size=100, shuffle=False)


# Validation and calculating BLEU score after fine-tuning
_, predictions = validate_and_calculate_bleu(sample_loader, encoder, decoder, vocab, device, False)
# Load the sample dataset
sample_dataset = pd.read_csv(sample_path)
image_names = sample_dataset['image'].tolist()  # Extracting the image names


output_path = "pred1a.csv"
with open(output_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['image_id', 'formula'])  
    for img_name, prediction in zip(image_names, predictions):
        writer.writerow([img_name, prediction])

print(f"Predictions written to {output_path}")
