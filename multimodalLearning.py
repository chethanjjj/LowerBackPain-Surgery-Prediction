import torch
import torch.nn as nn
import torch.nn.functional as F 

class MultimodalNetwork(nn.Module):

    # Parameters:
    #   numClasses (int): # of classes
    #   fcHiddenSize (int): hidden size for fully-connected layer
    #   gruHiddenSize (int): hidden size for gru layer
    #   gruInputSize (int): input size for each cell in the gru layer
    #   embeddingModel (word2vec model): pretrained word2vec model
    #   vocabSize (int): size of the vocabulary from the word2vec model
    #   embeddingDim (int): size of the embedding from the word2vec model (i.e. 300)
    #   numFilters (int): # of channels for CNN
    #   kernelSizes (list of ints): list of kernel sizes (height) (e.g. [3, 4, 5])
    #   staticInputSize (int): # of demographic features
    #   dropout (float): probability for a node in a single layer to be zero'd out during training
    #   freezeEmbeddings (boolean): indicates if the embeddings should not be updated during training
    # Exception: None
    # Purpose: initializes the multimodal deep learning architecture
    # Return: None
    def __init__(self,
                 numClasses,
                 fcHiddenSize,
                 timeInputSize,
                 gruHiddenSize,
                 gruInputSize,
                 embeddingModel,
                 vocabSize,
                 embeddingDim,
                 numFilters,
                 kernelSizes,
                 staticInputSize,
                 dropout=0.5,
                 freezeEmbeddings=False):
    
        super(MultimodalNetwork, self).__init__()


        # GRU
        self.gruEmbed = nn.Linear(in_features = timeInputSize,
                                  out_features = gruInputSize,
                                  bias = True)
        self.gru = nn.GRU(input_size=gruInputSize,
                           hidden_size=gruHiddenSize,
                           num_layers=1,
                           batch_first=True,
                           bias=True)
        
        # CNN

        ## EMBEDDING LAYER
        self.embedding = nn.Embedding(vocabSize, embeddingDim, padding_idx=0)
        self.embedding.weight = nn.Parameter(torch.from_numpy(embeddingModel.vectors))
        if freezeEmbeddings:
            self.embedding.requires_grad = False
        
        ## CONVOLUTION LAYER
        self.convs1D = nn.ModuleList([nn.Conv2d(1, numFilters, (k, embeddingDim), padding=(k-2, 0))
                                      for k in kernelSizes])

        # FULLY-CONNECTED
        totalNetworksOutput = gruHiddenSize + staticInputSize + len(kernelSizes)*numFilters
        self.seqFc = nn.Sequential(nn.Linear(totalNetworksOutput, fcHiddenSize),
                                   nn.ReLU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(fcHiddenSize, numClasses))
    
    # Parameters:
    #   x (tensor): the embeddings for the batch of documents with dimensions: (batchSize, 1, seqLength, embedding)
    #   conv (nn.Conv2d object): performs convolution on x
    # Exception: None
    # Purpose: for a given filter size, perform convolution then 1D max pooling
    # Return: 
    #   xMax (tensor): max-pooled results with size equal to the # of channels
    def convPool(self, x, conv):
        # convert (batchSize, channels, seqLength, 1) to (batchSize, channels, seqLength)
        x = F.relu(conv(x)).squeeze(3)

        # max pool then convert (batchSize, channels, 1) to (batchSize, channels)
        xMax = F.max_pool1d(x, x.size(2)).squeeze(2)

        return(xMax)
    
    # Parameters:
    #   textInput (tensor): text data, rows - patients, column - list of indices from pre-trainined word2vec model
    #   timeInput (tensor): temporal data, it's a 3D tensor with depth corresponding to patients, height corresponding to bins, and width corresponding to unique features
    #   staticInput (tensor): demographics data, rows - patients and columns - unique demographic features
    # Exception: None
    # Purpose: Predict the class for each patient using the deep learning
    # Return: 
    #   output (tensor): the predicted values for each class (columns) for all patients in the batch size (rows)
    def forward(self, textInput, timeInput, staticInput):

        # GRU
        gruInput = self.gruEmbed(timeInput)
        self.gru.flatten_parameters()
        gruOutput, gruHn = self.gru(gruInput)
        ## drop the layer dimension (i.e. the first dimension) since there is only 1 LSTM layer
        gruHn = gruHn.view(-1, self.gruHiddenSize)

        # CNN
        ## (batchSize, seqLength, embedding)
        embeds = self.embedding(textInput)
        ## insert 1 for channel dim: (batchSize, 1, seqLength, embedding)
        ##  this insert of 1 tells the convlution function that there is only 1 channel in the input
        embeds = embeds.unsqueeze(1)
        ## get output for each conv-pool layer
        convResults = [self.convPool(embeds, conv) for conv in self.convs1D]
        ## concat convPool results (batchSize, channels*len(kernelSizes))
        cnnOut = torch.cat(convResults, 1)

        # CONCAT
        networksOutput = torch.cat((gruHn, cnnOut, staticInput), 1)

        # FULLY-CONNECTED
        output = self.seqFc(networksOutput)

        return(output, networksOutput)

