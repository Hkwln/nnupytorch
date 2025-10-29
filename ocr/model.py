import torch
#here is the cnn model
# todo: do some research what kind of cnn structure is recomendet
    # BiLstm: Bidirectional Long Short-Term Memory Network
    # input_dimensions: height 128

class BiLstm(torch.nn.Modul):
    def __init__(self):

        super(cnn, self).__init__()
        self.input_layer = torch.nn.Linear()
        self.cnn_layer = torch.nn.LSTM(in_channels=3, out_channels= all_chars, kernel_size=16)
        self.flattening_layer = torch.nn.
        self.BiLstm_Layer = torch.nn.#?
        self.Dense_layer = torch.nn.Linear()

    def forward(x),
        b= self.pooling(torch.nn.functional.relu(self.convolutionlayer(x)))
        b = self.fc(b)
        out = b+x
        return out



