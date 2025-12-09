from conllu_token import Token
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense
from tensorflow.keras.models import Model
import numpy as np 
from typing import List, Tuple, Dict, Any

PAD = "<PAD>"
UNK = "<UNK>"
PAD_ID = 0
UNK_ID = 1
NONE_LABEL = "<NONE"


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
        self.pos_emb_dim = 25
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size

        ## Create dictionaries for mapping strings to IDs
        self.word2id: Dict[str, int] = {PAD: PAD_ID, UNK: UNK_ID}
        self.id2word: Dict[int, str] = {PAD_ID: PAD, UNK_ID: UNK}
        self.pos2id: Dict[str, int] = {PAD: PAD_ID, UNK: UNK_ID}
        self.id2pos: Dict[int, str] = {PAD_ID: PAD, UNK_ID: UNK}
        self.action2id: Dict[str, int] = {}
        self.id2action: Dict[int, str] = {}
        self.label2id: Dict[int, str] = {}
        self.id2label: Dict[int, str] = {}

        self.model: Model = None
        self.vocab_size = None
        self.pos_size = None

    def train(self, training_samples: list['Sample'], dev_samples: list['Sample']):
        """
        Trains the MLP model using the provided training and development samples.

        This method prepares the training data by mapping samples to IDs suitable for 
        embedding layers and then proceeds to compile and fit the Keras model.

        Parameters:
            training_samples (list[Sample]): A list of training samples for the parser.
            dev_samples (list[Sample]): A list of development samples used for model validation.
        """

        flat_train = []

        ## Sample objects extraction
        for sample in training_samples:
            if hasattr(sample, "state") and hasattr(sample, "transition"):
                flat_train.append(sample)
            else:
                try:
                    for inner in sample:
                        if hasattr(inner, "state") and hasattr(inner, "transition"):
                            flat_train.append(inner)
                except Exception:
                    pass
        
        ## Validation of samples
        if len(flat_train) == 0:
            raise ValueError("No valid Sample instances found in training_samples")
        
        flat_dev = []

        for sample in dev_samples:
            if hasattr(sample, "state") and hasattr(sample, "transition"):
                flat_dev.append(sample)
            else:
                try:
                    for inner in sample:
                        if hasattr(inner, "state") and hasattr(inner, "transition"):
                            flat_dev.append(inner)
                except Exception:
                    pass
        
        ## Vocabularies for words, pos taggs, actions and labels
        next_word_id = max(self.word2id.values()) + 1
        next_pos_id = max(self.pos2id.values()) + 1
        actions_set = set()
        labels_set = set([NONE_LABEL])

        ## Extract words and POS taggs from stack and buffer
        for sample in flat_train:
            state = sample.state
            S = getattr(state, "S")
            B = getattr(state, "B")

            ## Stack processing
            ## LIFO method - takes the last element and the element before the lest one
            for i in range(1,3):

                if len(S) >= i:
                    tok = S[-i] 
                else:
                    None

                w = None
                p = None
                if tok is not None:
                    w = getattr(tok, "form", None) or getattr(tok, "FORM", None) or getattr(tok, "word", None)
                    p = getattr(tok, "upos", None) or getattr(tok, "UPOS", None) or getattr(tok, "pos", None)

                if w is None:
                    w = PAD
                if p is None:
                    p = PAD

                ## Add word to vocabulary
                if w not in self.word2id:
                    self.word2id[w] = next_word_id
                    self.id2word[next_word_id] = w
                    next_word_id += 1
                if p not in self.pos2id:
                    self.pos2id[p] = next_pos_id
                    self.id2pos[next_pos_id] = p
                    next_pos_id += 1
            
            ## Buffer processing
            ## FIFO method - takes the first and second elements
            for i in range(0,2):
                tok = B[i] if len(B) > i else None
                w = None
                p = None
                if tok is not None:
                    w = getattr(tok, "form", None) or getattr(tok, "FORM", None) or getattr(tok, "word", None)
                    p = getattr(tok, "upos", None) or getattr(tok, "UPOS", None) or getattr(tok, "pos", None)
                
                if w is None:
                    w = PAD
                
                if p is None:
                    p = PAD
                
                if w not in self.word2id:
                    self.word2id[w] = next_word_id
                    self.id2word[next_word_id] = w
                    next_word_id += 1
                
                if p not in self.pos2id:
                    self.pos2id[p] = next_pos_id
                    self.id2pos[next_pos_id] = p
                    next_pos_id += 1

            ## Save actions and labels from smaples
            tr = sample.transition
            if tr and getattr(tr, "action", None) is not None:
                 action_name = tr.action 
            
            if tr and getattr(tr, "dependency", None) is not None:
                label_name = tr.dependency

            if action_name is not None:
                actions_set.add(action_name)

            if label_name is not None:
                labels_set.add(label_name)

        ## Calculate embedding dimension
        self.vocab_size = max(self.id2word.keys()) + 1
        self.pos_size = max(self.id2pos.keys()) + 1

        self.action2id = {a: i for i, a in enumerate(sorted(actions_set))}
        self.id2action = {i: a for a, i in self.action2id.items()}
        self.label2id = {l: i for i, l in enumerate(sorted(labels_set))}
        self.id2label = {i: l for l, i in self.label2id.items()}

        ## Data transfirmation for training_samples
        X_train = []
        y_action = []
        y_label = []

        ## Helper for extracting word ID
        def get_word_id(tok):
            if tok is None:
                return PAD_ID
            w = getattr(tok, "form", None) or getattr(tok, "FORM", None) or getattr(tok, "word", None)
            if w is None:
                return PAD_ID
            
            return self.word2id.get(w, UNK_ID)
        
        ## Helper for extracting POS ID
        def get_pos_id(tok):
            if tok is None:
                return PAD_ID
            p = getattr(tok, "upos", None) or getattr(tok, "UPOS", None) or getattr(tok, "pos", None)
            if p is None:
                return PAD_ID
            
            return self.pos2id.get(p, UNK_ID)
    
        for sample in flat_train:
            state = sample.state
            S = getattr(state, "S")
            B = getattr(state, "B")

            ## Construction of input for Dense layer
            s1 = get_word_id(S[-1]) if len(S) >= 1 else PAD_ID
            s2 = get_word_id(S[-2]) if len(S) >= 2 else PAD_ID
            b1 = get_word_id(B[0]) if len(B) >= 1 else PAD_ID
            b2 = get_word_id(B[1]) if len(B) >= 2 else PAD_ID
            ps1 = get_pos_id(S[-1]) if len(S) >= 1 else PAD_ID
            ps2 = get_pos_id(S[-2]) if len(S) >= 2 else PAD_ID
            pb1 = get_pos_id(B[0]) if len(B) >= 1 else PAD_ID
            pb2 = get_pos_id(B[1]) if len(B) >= 2 else PAD_ID

            X_train.append([s1, s2, b1, b2, ps1, ps2, pb1, pb2])

            tr = sample.transition

            if tr is None or getattr(tr, "action", None) is None:
                raise ValueError("Training sample has no transition.action")
            act_id = self.action2id[tr.action]

            if getattr(tr, "dependency", None) is not None:
                lbl = tr.dependency 
            else:
                NONE_LABEL

            lbl_id = self.label2id.get(lbl, self.label2id[NONE_LABEL])

            y_action.append(act_id)
            y_label.append(lbl_id)

        X_train = np.array(X_train, dtype='int32')
        y_action = np.array(y_action, dtype='int32')
        y_label = np.array(y_label, dtype='int32')

        ## Data transformation for dev_samples
        X_dev = []
        y_dev_action = []
        y_dev_label = []

        for sample in flat_dev:
            state = sample.state
            S = getattr(state, "S")
            B = getattr(state, "B")

            s1 = get_word_id(S[-1]) if len(S) >= 1 else PAD_ID
            s2 = get_word_id(S[-2]) if len(S) >= 2 else PAD_ID
            b1 = get_word_id(B[0]) if len(B) >= 1 else PAD_ID
            b2 = get_word_id(B[1]) if len(B) >= 2 else PAD_ID
            ps1 = get_pos_id(S[-1]) if len(S) >= 1 else PAD_ID
            ps2 = get_pos_id(S[-2]) if len(S) >= 2 else PAD_ID
            pb1 = get_pos_id(B[0]) if len(B) >= 1 else PAD_ID
            pb2 = get_pos_id(B[1]) if len(B) >= 2 else PAD_ID
           
            X_dev.append([s1, s2, b1, b2, ps1, ps2, pb1, pb2])

            tr = sample.transition
            if tr is None or getattr(tr, "action", None) is None:
                continue
            if tr.action not in self.action2id:
                continue
            act_id = self.action2id[tr.action]
            lbl = tr.dependency if getattr(tr, "dependency", None) is not None else NONE_LABEL
            if lbl not in self.label2id:
                lbl = NONE_LABEL
            lbl_id = self.label2id[lbl]

            y_dev_action.append(act_id)
            y_dev_label.append(lbl_id)

        X_dev = np.array(X_dev, dtype='int32') if X_dev else None
        if y_dev_action:
            y_dev_action = np.array(y_dev_action, dtype='int32')
        else:
            None
        if y_dev_label:
            y_dev_label = np.array(y_dev_label, dtype='int32') 
        else:
            None

        ## Construction of Keras
        input_ids = Input(shape=(8,), dtype='int32', name='input_ids')

        ## Split words and POS taggs
        words_slice = Lambda(lambda x: x[:, :4], output_shape=(4,))(input_ids)
        pos_slice = Lambda(lambda x: x[:, 4:], output_shape=(4,))(input_ids)

        ## Make embeddings
        word_emb = Embedding(input_dim=self.vocab_size, output_dim=self.word_emb_dim, input_length=4, name='word_emb')
        pos_emb = Embedding(input_dim=self.pos_size, output_dim=self.pos_emb_dim, input_length=4, name='pos_emb')

        ## Concatenation of embedded words and POS taggs
        w_embedded = Flatten()(word_emb(words_slice))
        p_embedded = Flatten()(pos_emb(pos_slice))

        concat = Concatenate()([w_embedded, p_embedded])

        ## Initialize hidden layer
        hidden = Dense(self.hidden_dim, activation='relu')(concat)

        ## Initialize output
        action_output = Dense(len(self.action2id), activation='softmax', name='action_output')(hidden)
        label_output = Dense(len(self.label2id), activation='softmax', name='label_output')(hidden)

        self.model = Model(inputs=input_ids, outputs=[action_output, label_output])

        ## Compile model
        self.model.compile(optimizer='adam', 
                           loss={
                               'action_output': 'sparse_categorical_crossentropy',
                               'label_output': 'sparse_categorical_crossentropy'
                           }, 
                           metrics={
                               'action_output': 'sparse_categorical_accuracy',
                               'label_output': 'sparse_categorical_accuracy'
                           })
        
        if X_dev is not None:
            val = (X_dev, {'action_output': y_dev_action, 'label_output': y_dev_label})
        else:
            val = None

        ## Train model
        self.model.fit(
            X_train, 
            {'action_output': y_action, 'label_output': y_label},
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=val,
            verbose=2)
        

    def evaluate(self, samples: list['Sample']):
        """
        Evaluates the model's performance on a set of samples.

        This method is used to assess the accuracy of the model in predicting the correct
        transition and dependency types. The expected accuracy range is between 75% and 85%.

        Parameters:
            samples (list[Sample]): A list of samples to evaluate the model's performance.
        """
        flat = []

        ## Sample objects extraction
        for s in samples:
            if hasattr(s, "state") and hasattr(s, "transition"):
                flat.append(s)
            else:
                try:
                    for inner in s:
                        if hasattr(inner, "state") and hasattr(inner, "transition"):
                            flat.append(inner)
                except Exception:
                    pass
        
        if len(flat) == 0:
            return 0.0, 0.0


        ## Build X - for data evaluation 
        X_eval = []
        y_action_true = []
        y_label_true = []

        ## Helper functions
        def get_word_id(tok):
            if tok is None:
                return PAD_ID
            w = getattr(tok, "form", None) or getattr(tok, "FORM", None) or getattr(tok, "word", None)
            if w is None:
                return PAD_ID
            return self.word2id.get(w, UNK_ID)

        def get_pos_id(tok):
            if tok is None:
                return PAD_ID
            p = getattr(tok, "upos", None) or getattr(tok, "UPOS", None) or getattr(tok, "pos", None)
            if p is None:
                return PAD_ID
            return self.pos2id.get(p, UNK_ID)
        
        for sample in samples:
            
            state = sample.state
            S = getattr(state, "S")
            B = getattr(state, "B")

            ## Create feature array
            s1 = get_word_id(S[-1]) if len(S) >= 1 else PAD_ID
            s2 = get_word_id(S[-2]) if len(S) >= 2 else PAD_ID
            b1 = get_word_id(B[0]) if len(B) >= 1 else PAD_ID
            b2 = get_word_id(B[1]) if len(B) >= 2 else PAD_ID
            ps1 = get_pos_id(S[-1]) if len(S) >= 1 else PAD_ID
            ps2 = get_pos_id(S[-2]) if len(S) >= 2 else PAD_ID
            pb1 = get_pos_id(B[0]) if len(B) >= 1 else PAD_ID
            pb2 = get_pos_id(B[1]) if len(B) >= 2 else PAD_ID

            X_eval.append([s1, s2, b1, b2, ps1, ps2, pb1, pb2])

            ## Construction of true labels
            tr = sample.transition
            y_action_true.append(self.action2id[tr.action])
            
            if getattr(tr, "dependency", None) is not None:
                lbl = tr.dependency 
            else:
                NONE_LABEL

            y_label_true.append(self.label2id.get(lbl, self.label2id[NONE_LABEL]))

        X_eval = np.array(X_eval, dtype='int32')
        y_action_true = np.array(y_action_true, dtype='int32')
        y_label_true = np.array(y_label_true, dtype='int32')

        ## Predict and evaluate
        pred_action_probs, pred_label_probs = self.model.predict(X_eval, verbose=0)

        pred_action = np.argmax(pred_action_probs, axis=1)
        pred_label = np.argmax(pred_label_probs, axis=1)

        action_accuracy = float(np.mean(pred_action == y_action_true))
        label_accuracy = float(np.mean(pred_label == y_label_true))

        return action_accuracy, label_accuracy
    
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


## TODO: Debugging - Kriszta
     
if __name__ == "__main__":
    
    model = ParserMLP()
    print("ParserMLP ready. Use train/evaluate/run from your pipeline.")