def fprop(input_batch, embed_weights, embed_to_hid_weights, hid_to_output_weights, hid_bias, output_bias):
    """
    This method forward propagates through a neural network. Inputs:
    input_batch: The input data as a matrix of size numwords X batchsize where, numwords is the number of words, batchsize is the number of data points. So, if input_batch(i, j) = k then the ith word in data point j is word index k of the vocabulary.

    word_embedding_weights: Word embedding as a matrix of size vocab_size X numhid1, where vocab_size is the size of the vocabulary, numhid1 is the dimensionality of the embedding space.

    embed_to_hid_weights: Weights between the word embedding layer and hidden layer as a matrix of size numhid1*numwords X numhid2, numhid2 is the number of hidden units.

    hid_to_output_weights: Weights between the hidden layer and output softmax unit as a matrix of size numhid2 X vocab_size

    hid_bias: Bias of the hidden layer as a matrix of size numhid2 X 1.

    output_bias: Bias of the output layer as a matrix of size vocab_size X 1.

    Outputs:
    embedding_layer_state: State of units in the embedding layer as a matrix of size numhid1*numwords X batchsize

    hidden_layer_state: State of units in the hidden layer as a matrix of size numhid2 X batchsize

    output_layer_state: State of units in the output layer as a matrix of size vocab_size X batchsize
    """
    batchsize, numwords = input_batch.shape
    numhid1, vocabsize = embed_weights.shape

