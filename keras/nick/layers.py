from keras.layers.embeddings import Embedding

class ImmutableEmbedding(Embedding):
    '''
        Same as keras.layers.Embedding except the weights are parameters
        of the network.  This can be useful when the layer is initialized
        with pre-trained embeddings, such as Word2Vec.

        @input_dim: size of vocabulary (highest input integer + 1)
        @out_dim: size of dense representation
    '''
    def __init__(self, input_dim, output_dim, **kwargs):

        super(ImmutableEmbedding, self).__init__(
                input_dim, output_dim, **kwargs)

        print("W", self.get_weights())
        print("params", self.params)
        self.params = []
        print("params", self.params)
