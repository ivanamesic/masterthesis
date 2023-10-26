import torch.nn as nn

# This is a model consisting of 3 components: an entry LSTM layer called encoder, an attention layer and an output LSTM layer called the decoder.
class LSTMMultiLayerWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, num_heads, output_dim):
        super(LSTMMultiLayerWithAttention, self).__init__()

        self.encoder = nn.LSTM(input_dim, hidden_dim1, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_dim2, num_heads)
        self.fc = nn.Linear(hidden_dim2, output_dim)

    def forward(self, input_seq):
        # Pass the input sequence through the encoder LSTM
        encoder_output, _ = self.encoder(input_seq)

        # Pass the encoder output through the decoder LSTM
        decoder_output, _ = self.decoder(encoder_output)

        # Transpose the decoder output to match the shape expected by MultiheadAttention
        decoder_output = decoder_output.permute(1, 0, 2)

        # Apply the Multihead Attention mechanism
        attn_output, _ = self.attention(decoder_output, decoder_output, decoder_output)

        # Transpose the attention output back to the original shape
        attn_output = attn_output.permute(1, 0, 2)

        # Apply the linear layer to get the final output
        output = self.fc(attn_output[:, -1, :])

        return output
    
# This is a simple benchmark model consisting of only 1 LSTM layer and a linear fully connected layer
# Inputs to the constructor function are the hyperparameters of the model:
#   - input size: number of features in each sample of input data
#   - hidden size: number of hidden layer neurons
#   - output size: output dimension
class LSTMSimple(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMSimple, self).__init__()
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # Linear fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Forward pass through the LSTM layer
        lstm_out, _ = self.lstm(x)
        
        # Get the output from the last time step
        last_output = lstm_out[:, -1, :]
        
        # Pass the last output through the fully connected (output) layer
        prediction = self.fc(last_output)
        
        return prediction

