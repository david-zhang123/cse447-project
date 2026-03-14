

if __name__ == "__main__":
    pred_path = 'output/pred.txt'
    correct_path = 'output/correct_next_char.txt'

    with open(pred_path) as f:
        preds = [line.strip() for line in f]
    with open(correct_path) as f:
        corrects = [line.strip() for line in f]
    correct_count = 0
    for pred, correct in zip(preds, corrects):
        if correct in pred or (correct == '' and len(pred) == 2):
            correct_count += 1
        else:
            print(f"Incorrect prediction: '{pred.strip()}' for correct char '{correct.strip()}'")
    print(f"Accuracy: {correct_count / len(corrects):.6f}")