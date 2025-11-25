from conllu_token import Token
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense
from tensorflow.keras.models import Model
import numpy as np 


class ParserMLP:
    """
    A Multi-Layer Perceptron (MLP) class for a dependency parser, using TensorFlow and Keras.

    This class implements a neural network model designed to predict transitions in a dependency 
    parser. It utilizes the Keras Functional API, which is more suited for multi-task learning scenarios 
    like this one. The network is trained to map parsing states to transition actions, facilitating 
    the parsing process in natural language processing tasks.

    Attributes:
        word_emb_dim (int): Dimensionality of the word embeddings. Defaults to 100.
        hidden_dim (int): Dimension of the hidden layer in the neural network. Defaults to 64.
        epochs (int): Number of training epochs. Defaults to 1.
        batch_size (int): Size of the batches used in training. Defaults to 64.

    Methods:
        train(training_samples, dev_samples): Trains the MLP model using the provided training and 
            development samples. It maps these samples to IDs that can be processed by an embedding 
            layer and then calls the Keras compile and fit functions.

        evaluate(samples): Evaluates the performance of the model on a given set of samples. The 
            method aims to assess the accuracy in predicting both the transition and dependency types, 
            with expected accuracies ranging between 75% and 85%.

        run(sents): Processes a list of sentences (tokens) using the trained model to perform dependency 
            parsing. This method implements the vertical processing of sentences to predict parser 
            transitions for each token.

        Feel free to add other parameters and functions you might need to create your model
    """

    def __init__(self, word_emb_dim: int = 100, hidden_dim: int = 64, 
                 epochs: int = 1, batch_size: int = 64):
        """
        Initializes the ParserMLP class with the specified dimensions and training parameters.

        Parameters:
            word_emb_dim (int): The dimensionality of the word embeddings.
            hidden_dim (int): The size of the hidden layer in the MLP.
            epochs (int): The number of epochs for training the model.
            batch_size (int): The batch size used during model training.
        """
        self.word_emb_dim = word_emb_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size

        self.word2id = {}
        self.id2word = {}
        self.action2id = {}
        self.id2action = {}
        self.label2id = {}
        self.id2label = {}

        self.model = None

    
    def train(self, training_samples: list['Sample'], dev_samples: list['Sample']):
        """
        Trains the MLP model using the provided training and development samples.

        This method prepares the training data by mapping samples to IDs suitable for 
        embedding layers and then proceeds to compile and fit the Keras model.

        Parameters:
            training_samples (list[Sample]): A list of training samples for the parser.
            dev_samples (list[Sample]): A list of development samples used for model validation.
        """

        # Construction of vocabulary
        word2id = {}
        id2word = {}
        idx = 1

        for sample in training_samples:
            for token in sample.tokens:
                if token.word not in word2id:
                    word2id[token.word] = idx
                    id2word[idx] = token.word
                    idx += 1

        vocab_size = len(word2id) + 1

        # Construction of action2id and label2id
        self.action2id = {}
        self.id2action = {}
        idx = 0
        for sample in training_samples:
            for action in sample.actions:
                if action not in self.action2id:
                    self.action2id[action] = idx
                    self.id2action[idx] = action
                    idx += 1

        self.label2id = {}
        self.id2label= {}
        idx = 0
        for sample in training_samples:
            for label in sample.labels:
                if label not in self.label2id:
                    self.label2id[label] = idx
                    self.id2label[idx] = label
                    idx += 1

        self.word2id = word2id
        self.id2word = id2word

        # Data transfirmation for training_samples
        X_train = []
        y_action = []
        y_label = []

        for sample in training_samples:
            for step in sample.steps:
                stack = step.stack
                buffer = step.buffer

                # Extracting first 2 elements or padding
                s1 = self.word2id.get(stack[0].word, 0) if len(stack) > 0 else 0
                s2 = self.word2id.get(stack[1].word, 0) if len(stack) > 1 else 0
                b1 = self.word2id.get(buffer[0].word, 0) if len(buffer) > 0 else 0
                b2 = self.word2id.get(buffer[1].word, 0) if len(buffer) > 1 else 0

                x = [s1, s2, b1, b2]

                # One hot encoding for action and label
                y_a = np.zeros(len(self.action2id))
                y_a[self.action2id[step.action]] = 1

                y_l = np.zeros(len(self.label2id))
                y_l[self.label2id[step.label]] = 1

                # Add to list
                X_train.append(x)
                y_action.append(y_a)
                y_label.append(y_l)

        # Data transformation for dev_samples
        X_dev = []
        y_dev_action = []
        y_dev_label = []

        for sample in dev_samples:
            for step in sample.steps:
                stack = step.stack
                buffer = step.buffer

                s1 = self.word2id.get(stack[0].word, 0) if len(stack) > 0 else 0
                s2 = self.word2id.get(stack[1].word, 0) if len(stack) > 1 else 0
                b1 = self.word2id.get(buffer[0].word, 0) if len(buffer) > 0 else 0
                b2 = self.word2id.get(buffer[1].word, 0) if len(buffer) > 1 else 0

                x = [s1, s2, b1, b2]

                y_a = np.zeros(len(self.action2id))
                y_a[self.action2id[step.action]] = 1

                y_l = np.zeros(len(self.label2id))
                y_l[self.label2id[step.label]] = 1

                X_dev.append(x)
                y_dev_action.append(y_a)
                y_dev_label.append(y_l)
        
        # Convert to numpy array
        X_train = np.array(X_train)
        y_action = np.array(y_action)
        y_label = np.array(y_label)
        X_dev = np.array(X_dev)
        y_dev_action = np.array(y_dev_action)
        y_dev_label = np.array(y_dev_label)

           
        
        input_ids = Input(shape=(4,), dtype='int32', name='input_ids')

        embedding_layer = Embedding(input_dim=vocab_size, output_dim=self.word_emb_dim, input_length=4)
        embedded = embedding_layer(input_ids)
        flattened = Flatten()(embedded)

        hidden = Dense(self.hidden_dim, activation='relu')(flattened)

        action_output = Dense(len(self.action2id), activation='softmax', name='action_output')(hidden)
        label_output = Dense(len(self.label2id), activation= 'softmax', name='label_output')(hidden)

        self.model = Model(inputs=input_ids, outputs=[action_output, label_output])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        self.model.fit(X_train, [y_action, y_label], 
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       validation_data=(X_dev, [y_dev_action, y_dev_label])
                       )
        

    def evaluate(self, samples: list['Sample']):
        """
        Evaluates the model's performance on a set of samples.

        This method is used to assess the accuracy of the model in predicting the correct
        transition and dependency types. The expected accuracy range is between 75% and 85%.

        Parameters:
            samples (list[Sample]): A list of samples to evaluate the model's performance.
        """
        raise NotImplementedError
    
    def run(self, sents: list['Token']):
        """
        Executes the model on a list of sentences to perform dependency parsing.

        This method implements the vertical processing of sentences, predicting parser 
        transitions for each token in the sentences.

        Parameters:
            sents (list[Token]): A list of sentences, where each sentence is represented 
                                 as a list of Token objects.
        """

        # Main Steps for Processing Sentences:
        # 1. Initialize: Create the initial state for each sentence.
        # 2. Feature Representation: Convert states to their corresponding list of features.
        # 3. Model Prediction: Use the model to predict the next transition and dependency type for all current states.
        # 4. Transition Sorting: For each prediction, sort the transitions by likelihood using numpy.argsort, 
        #    and select the most likely dependency type with argmax.
        # 5. Validation Check: Verify if the selected transition is valid for each prediction. If not, select the next most likely one.
        # 6. State Update: Apply the selected actions to update all states, and create a list of new states.
        # 7. Final State Check: Remove sentences that have reached a final state.
        # 8. Iterative Process: Repeat steps 2 to 7 until all sentences have reached their final state.


        raise NotImplementedError


if __name__ == "__main__":
    
    model = ParserMLP()