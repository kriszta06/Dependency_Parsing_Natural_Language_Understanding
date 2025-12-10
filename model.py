from conllu_token import Token
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Lambda, Concatenate
from tensorflow.keras.models import Model
from state import State
import numpy as np 
from typing import List, Tuple, Dict, Any
from algorithm import Sample
np.random.seed(42)
tf.random.set_seed(42)


PAD = "<PAD>"
UNK = "<UNK>"
PAD_ID = 0
UNK_ID = 1
NONE_LABEL = "<NONE>"

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
        print("s a lucrat")
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

            if tr is None or getattr(tr, "action", None) is None:
                continue
            action_name = tr.action

            if getattr(tr, "dependency", None) is not None:
                label_name = tr.dependency
            else:
                label_name = NONE_LABEL


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
               continue

            act_id = self.action2id[tr.action]

            if getattr(tr, "dependency", None) is not None:
                lbl = tr.dependency 
            else:
                lbl = NONE_LABEL

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

        X_dev = np.array(X_dev, dtype='int32') if len(X_dev) > 0 else None

        if y_dev_action:
            y_dev_action = np.array(y_dev_action, dtype='int32')

        if y_dev_label:
            y_dev_label = np.array(y_dev_label, dtype='int32') 

        print(f"Processed {len(flat_train)} training samples, {len(flat_dev)} dev samples")

        print(f"Vocab size: {self.vocab_size}, POS size: {self.pos_size}")
        print(f"Number of actions: {len(self.action2id)}, Number of labels: {len(self.label2id)}")
        print(f"Number of training samples: {len(X_train)}, number of dev samples: {X_dev.shape[0] if X_dev is not None else 0}")

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


        flat = []

        # 1. Aplatizăm lista (pentru că oracle() întoarce listă de Sample per propoziție)
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

        # 2. Construim X_eval și label-urile adevărate
        X_eval = []
        y_action_true = []
        y_label_true = []

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

        # ⚠️ AICI folosim flat, nu samples
        for sample in flat:

            state = sample.state
            S = getattr(state, "S")
            B = getattr(state, "B")

            # feature vector pentru eșantionul curent
            s1 = get_word_id(S[-1]) if len(S) >= 1 else PAD_ID
            s2 = get_word_id(S[-2]) if len(S) >= 2 else PAD_ID
            b1 = get_word_id(B[0]) if len(B) >= 1 else PAD_ID
            b2 = get_word_id(B[1]) if len(B) >= 2 else PAD_ID
            ps1 = get_pos_id(S[-1]) if len(S) >= 1 else PAD_ID
            ps2 = get_pos_id(S[-2]) if len(S) >= 2 else PAD_ID
            pb1 = get_pos_id(B[0]) if len(B) >= 1 else PAD_ID
            pb2 = get_pos_id(B[1]) if len(B) >= 2 else PAD_ID

            tr = sample.transition
            if tr is None or getattr(tr, "action", None) is None:
                continue  # dacă nu avem acțiune, nu evaluăm sample-ul

            # adăugăm features DOAR dacă avem și labeluri
            X_eval.append([s1, s2, b1, b2, ps1, ps2, pb1, pb2])

            y_action_true.append(self.action2id[tr.action])

            lbl = tr.dependency if getattr(tr, "dependency", None) is not None else NONE_LABEL
            if lbl not in self.label2id:
                lbl = NONE_LABEL
            y_label_true.append(self.label2id.get(lbl, self.label2id[NONE_LABEL]))

        if len(X_eval) == 0:
            return 0.0, 0.0

        X_eval = np.array(X_eval, dtype='int32')
        y_action_true = np.array(y_action_true, dtype='int32')
        y_label_true = np.array(y_label_true, dtype='int32')

        # 3. Predict & evaluate
        pred_action_probs, pred_label_probs = self.model.predict(X_eval, batch_size=64, verbose=0)

        pred_action = np.argmax(pred_action_probs, axis=1)
        pred_label = np.argmax(pred_label_probs, axis=1)

        action_accuracy = float(np.mean(pred_action == y_action_true))
        label_accuracy = float(np.mean(pred_label == y_label_true))

        print(f"Evaluated {len(X_eval)} samples")
        print(f"Action accuracy: {action_accuracy}, Label accuracy: {label_accuracy}")

        return action_accuracy, label_accuracy

    # Helper method to extract features from a state
    def extract_features(self, state):
        
        S = getattr(state, "S")
        B = getattr(state, "B")

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
        
        s1 = get_word_id(S[-1]) if len(S) >= 1 else PAD_ID
        s2 = get_word_id(S[-2]) if len(S) >= 2 else PAD_ID
        b1 = get_word_id(B[0]) if len(B) >= 1 else PAD_ID
        b2 = get_word_id(B[1]) if len(B) >= 2 else PAD_ID
        ps1 = get_pos_id(S[-1]) if len(S) >= 1 else PAD_ID
        ps2 = get_pos_id(S[-2]) if len(S) >= 2 else PAD_ID
        pb1 = get_pos_id(B[0]) if len(B) >= 1 else PAD_ID
        pb2 = get_pos_id(B[1]) if len(B) >= 2 else PAD_ID

        return [s1, s2, b1, b2, ps1, ps2, pb1, pb2]
    


    def is_valid_transition(self, state, action):

        if action == "SHIFT" and len(state.B) > 0:
            return True
        if action in ("LEFT-ARC", "RIGHT-ARC") and len(state.S) >= 2:
            return True
        
        return False
    
    def apply_transition(self, state, action, dep_label):
        S = state.S.copy()
        B = state.B.copy()
        A = state.A.copy()

        # --- SHIFT ---
        if action == "SHIFT":
            if len(B) == 0:
                return state  # invalid SHIFT → nu facem nimic
            S.append(B.pop(0))
            return State(s=S, b=B, a=A)

        # --- LEFT-ARC ---
        if action == "LEFT-ARC":
            if len(S) < 2:
                return state  # nu putem aplica
            head = S[-1]
            dep  = S[-2]

            # blocăm cazuri invalide (ex: root ca dependent)
            if dep.id <= 0 or head.id <= 0:
                return state

            A.add((head.id, dep_label, dep.id))
            S.pop(-2)
            return State(s=S, b=B, a=A)

        # --- RIGHT-ARC ---
        if action == "RIGHT-ARC":
            if len(S) < 2:
                return state
            head = S[-2]
            dep  = S[-1]

            if dep.id <= 0 or head.id <= 0:
                return state

            A.add((head.id, dep_label, dep.id))
            S.pop(-1)
            return State(s=S, b=B, a=A)

        # --- REDUCE ---
        if action == "REDUCE":
            if len(S) == 0:
                return state
            # doar reducerea dacă tokenul are head deja atribuit
            top = S[-1]
            has_head = any(dep == top.id for (_, _, dep) in A)
            if has_head:
                S.pop()
            return State(s=S, b=B, a=A)

        return state

    def transition_from_index(self, idx):
        return self.id2action[idx]
    
    def is_final(self, state):
        return len(state.B) == 0 and len(state.S) == 1
    
    def predict_transitions(self, sentence):
        
        """
        Predicts the full transition sequence for a given sentence.
        Returns a list of (action, label) pairs.
        """
        from algorithm import State  # ensure State is imported

        # Initialize state
        state = State(s=[sentence[0]], b=sentence[1:], a=set())

        transitions = []

        while not self.is_final(state):

            # Extract features for current state
            feats = np.array([self.extract_features(state)], dtype="int32")

            # Predict action + dependency
            action_scores, dep_scores = self.model.predict(feats, verbose=0)

            sorted_actions = action_scores[0].argsort()[::-1]

            chosen_action = None
            chosen_label = None

            for a_idx in sorted_actions:
                action = self.id2action[a_idx]
                label_idx = np.argmax(dep_scores[0])
                label = self.id2label[label_idx]

                if self.is_valid_transition(state, action):
                    chosen_action = action
                    chosen_label = label
                    break

            if chosen_action is None:
                break  # dead state, avoid infinite loop

            transitions.append((chosen_action, chosen_label))

            # Apply transition to update state
            state = self.apply_transition(state, chosen_action, chosen_label)

        return transitions
    
    def transitions_to_tree(self, sentence, transitions):
        """
        Applies a predicted transition sequence to reconstruct the dependency tree.
        Returns the final State object (state.A conține arcele).
        """

        from algorithm import State
        
        # Start from initial parser state
        state = State(s=[sentence[0]], b=sentence[1:], a=set())

        for (action, label) in transitions:
            if not self.is_valid_transition(state, action):
                continue  # ignore illegal transitions
            state = self.apply_transition(state, action, label)

            if self.is_final(state):
                break

        return state

    def run(self, sentences):
        """
        Runs parsing on a list of sentences using predicted transitions.
        Returns final state (with arcs) for each sentence.
        """
        final_states = []

        for sent in sentences:
            transitions = self.predict_transitions(sent)
            final_state = self.transitions_to_tree(sent, transitions)
            final_states.append(final_state)

        return final_states


    
 #   def run(self, sents: list['Token']):
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
        states = [State(s=[sent[0]], b=sent[1:], a=set()) for sent in sents]
        
        final_states = []

        while states:

            # 2. Feature Representation: Convert states to their corresponding list of features.
            features = np.array([self.extract_features(state) for state in states], dtype='int32')


            # 3. Model Prediction: Use the model to predict the next transition and dependency type for all current states.
            transitions_scores, dep_scores = self.model.predict(features, verbose=0)

            new_states = []

            # 4. Transition Sorting: For each prediction, sort the transitions by likelihood using numpy.argsort, 
            #    and select the most likely dependency type with argmax.
            # 5. Validation Check: Verify if the selected transition is valid for each prediction. If not, select the next most likely one.
            for i, state in enumerate(states):

                sorted_transitions = transitions_scores[i].argsort()[::-1]
                selected_transition = None
                selected_dep = None

                for t_idx in sorted_transitions:
                    t = self.transition_from_index(t_idx)
                    dep_label_idx = np.argmax(dep_scores[i])
                    dep_label = self.id2label[dep_label_idx]

                    if self.is_valid_transition(state, t):
                        selected_transition = t
                        selected_dep = dep_label
                        break

                if selected_transition is None:
                    continue
                
                 # 6. State Update: Apply the selected actions to update all states, and create a list of new states.
                state = self.apply_transition(state, selected_transition, selected_dep)

                 # 7. Final State Check: Remove sentences that have reached a final state.
                if self.is_final(state):
                    final_states.append(state)
                else:
                    new_states.append(state)

             # 8. Iterative Process: Repeat steps 2 to 7 until all sentences have reached their final state.
            states = new_states

        return final_states

        
        
       
       


     
if __name__ == "__main__":
    
    model = ParserMLP()
    print("ParserMLP ready. Use train/evaluate/run from your pipeline.")