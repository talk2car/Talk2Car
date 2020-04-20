import torch
import torch.nn as nn

class TextEncoder(torch.nn.Module):
    def __init__(self, hidden_size, input_dim, n_layers=1, dropout=0):
        super(TextEncoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(input_dim, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, n_layers,
                      dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
    
    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        
        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        
        # Unpack padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        
        # We will use the sum of the final hidden state of the backward and foward pass. 
        hidden = torch.sum(hidden, dim = 0)
       
        return outputs, hidden
