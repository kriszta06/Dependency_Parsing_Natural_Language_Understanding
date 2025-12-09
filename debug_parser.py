from model import ParserMLP, State
import numpy as np

# ---------- Fake structures for testing ----------
class FakeToken:
    def __init__(self, form, upos, id):
        self.form = form
        self.upos = upos
        self.id = id  # needed by apply_transition

class FakeState:
    def __init__(self, S, B, A=None):
        self.S = S
        self.B = B
        self.A = A or set()

class FakeTransition:
    def __init__(self, action, dependency):
        self.action = action
        self.dependency = dependency

class FakeSample:
    def __init__(self, state, transition):
        self.state = state
        self.transition = transition

# ---------- Build fake data ----------
t1 = FakeToken("John", "PROPN", 1)
t2 = FakeToken("loves", "VERB", 2)
t3 = FakeToken("Mary", "PROPN", 3)

# Sample 1: initial state
state1 = FakeState(S=[t1], B=[t2, t3])
tr1 = FakeTransition(action="SHIFT", dependency="<NONE>")
samp1 = FakeSample(state1, tr1)

# Sample 2: after shifting "loves"
state2 = FakeState(S=[t1, t2], B=[t3])
tr2 = FakeTransition(action="LEFT-ARC", dependency="nsubj")
samp2 = FakeSample(state2, tr2)

train_data = [samp1, samp2]

# ---------- Initialize and train ParserMLP ----------
model = ParserMLP(epochs=5, hidden_dim=32)  # small hidden_dim for fast test
model.train(train_data, dev_samples=train_data)

# ---------- Print vocabularies ----------
print("\n--- Vocabularies ---")
print("word2id:", model.word2id)
print("pos2id:", model.pos2id)
print("action2id:", model.action2id)
print("label2id:", model.label2id)

# ---------- Evaluate on training data ----------
accA, accL = model.evaluate(train_data)
print("\nAccuracy action:", accA)
print("Accuracy label:", accL)

# ---------- Check input features ----------
print("\n--- Debug: Input features ---")
def print_features(sample):
    state = sample.state
    S = state.S
    B = state.B
    s1 = model.word2id.get(S[-1].form, 1) if len(S)>=1 else 0
    s2 = model.word2id.get(S[-2].form, 1) if len(S)>=2 else 0
    b1 = model.word2id.get(B[0].form, 1) if len(B)>=1 else 0
    b2 = model.word2id.get(B[1].form, 1) if len(B)>=2 else 0
    ps1 = model.pos2id.get(S[-1].upos, 1) if len(S)>=1 else 0
    ps2 = model.pos2id.get(S[-2].upos, 1) if len(S)>=2 else 0
    pb1 = model.pos2id.get(B[0].upos, 1) if len(B)>=1 else 0
    pb2 = model.pos2id.get(B[1].upos, 1) if len(B)>=2 else 0
    print(f"Features: {[s1, s2, b1, b2, ps1, ps2, pb1, pb2]}")

for s in train_data:
    print_features(s)

# ---------- Print embedding vectors ----------
print("\n--- Debug: Embedding vectors ---")
word_emb_weights = model.model.get_layer("word_emb").get_weights()[0]
pos_emb_weights = model.model.get_layer("pos_emb").get_weights()[0]

for w in ["John", "loves", "Mary"]:
    wid = model.word2id[w]
    print(f"Embedding for '{w}': {word_emb_weights[wid]}")

for p in ["PROPN", "VERB"]:
    pid = model.pos2id[p]
    print(f"Embedding for POS '{p}': {pos_emb_weights[pid]}")

# ---------- Prepare sentences for run() ----------
sentences = [[t1, t2, t3]]

# ---------- Run parser ----------
final_states = model.run(sentences)

print("\n--- Final states after run() ---")
for i, state in enumerate(final_states):
    print(f"Sentence {i+1}:")
    print("Stack:", [tok.form for tok in state.S])
    print("Buffer:", [tok.form for tok in state.B])
    print("Arcs:", state.A)

# ---------- Manual prediction ----------
print("\n--- Manual prediction ---")
feat = np.array([[model.word2id.get("John", 1),
                  model.word2id.get("loves", 1),
                  model.word2id.get("Mary", 1),
                  0,  # PAD
                  model.pos2id.get("PROPN", 1),
                  model.pos2id.get("VERB", 1),
                  model.pos2id.get("PROPN", 1),
                  0]], dtype='int32')

predA, predL = model.model.predict(feat)
predA_id = np.argmax(predA, axis=1)
predL_id = np.argmax(predL, axis=1)

print("Predicted action probabilities:", predA)
print("Predicted label probabilities:", predL)
print("Predicted action ID:", predA_id)
print("Mapped action:", [model.id2action[i] for i in predA_id])
print("Predicted label ID:", predL_id)
print("Mapped label:", [model.id2label[i] for i in predL_id])
