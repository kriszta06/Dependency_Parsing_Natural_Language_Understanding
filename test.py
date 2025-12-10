# inspect_prediction.py

def read_conllu(path):
    sentences = []
    sent = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if sent:
                    sentences.append(sent)
                    sent = []
                continue

            parts = line.split("\t")
            if len(parts) < 8:
                continue

            token_id = int(parts[0])
            head = int(parts[6]) if parts[6].isdigit() else -1
            deprel = parts[7]

            sent.append((token_id, head, deprel))

    return sentences


def inspect_sentences(sentences):
    for i, sent in enumerate(sentences):
        print(f"\n=== Sentence {i+1} ===")

        seen_deps = set()
        duplicate_deps = []
        invalid_heads = []
        roots = []

        max_id = len(sent)

        for (tid, head, dep) in sent:
            # 1) Duplicate dependent?
            if tid in seen_deps:
                duplicate_deps.append(tid)
            seen_deps.add(tid)

            # 2) Invalid head index?
            if head < 0 or head > max_id:
                invalid_heads.append((tid, head))

            # 3) Multiple roots?
            if head == 0:
                roots.append(tid)

        if duplicate_deps:
            print("❌ Duplicate dependent IDs found:", duplicate_deps)
        else:
            print("✔ No duplicate dependents")

        if invalid_heads:
            print("❌ Invalid HEAD values:", invalid_heads)
        else:
            print("✔ All HEAD indices valid")

        if len(roots) == 0:
            print("❌ No ROOT assigned in this sentence")
        elif len(roots) > 1:
            print("❌ Multiple roots:", roots)
        else:
            print("✔ Single root:", roots[0])


def main():
    path = "Output/prediction.conllu"  # <-- modify if needed
    print("Reading:", path)
    sentences = read_conllu(path)
    print(f"Loaded {len(sentences)} sentences")
    inspect_sentences(sentences)


if __name__ == "__main__":
    main()
