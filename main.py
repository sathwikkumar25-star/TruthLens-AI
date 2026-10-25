
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix

MODEL_NAME = "FacebookAI/roberta-large-mnli"

print("Loading TRUTHLENS AI evaluation engine...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

id2label = model.config.id2label

# Convert model labels to consistent names.
label_map = {
    "entailment": "entailment",
    "contradiction": "contradiction",
    "neutral": "neutral"
}


def verify_claim(evidence, claim):
    inputs = tokenizer(
        evidence,
        claim,
        return_tensors="pt",
        truncation=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0]

    scores = {
        label_map[id2label[i].lower()]: probabilities[i].item()
        for i in range(len(probabilities))
    }

    predicted_label = max(scores, key=scores.get)

    return predicted_label, scores


def evaluate_model():
    test_cases = [
        {
            "evidence": "Mercury is the closest planet to the Sun.",
            "claim": "Mercury is the nearest planet to the Sun.",
            "expected": "entailment"
        },
        {
            "evidence": "Mercury is the closest planet to the Sun.",
            "claim": "Venus is the closest planet to the Sun.",
            "expected": "contradiction"
        },
        {
            "evidence": "Mercury is the closest planet to the Sun.",
            "claim": "Mercury has two moons.",
            "expected": "neutral"
        },
        {
            "evidence": "Water freezes at 0 degrees Celsius under standard atmospheric pressure.",
            "claim": "Water freezes at 0 degrees Celsius under standard atmospheric pressure.",
            "expected": "entailment"
        },
        {
            "evidence": "The Earth revolves around the Sun.",
            "claim": "The Sun revolves around the Earth.",
            "expected": "contradiction"
        },
        {
            "evidence": "Python was first released in 1991.",
            "claim": "Python was first released in 2005.",
            "expected": "contradiction"
        }
    ]

    expected_labels = []
    predicted_labels = []

    print("\n=== TRUTHLENS AI v6 ===")

    for index, case in enumerate(test_cases, start=1):
        prediction, scores = verify_claim(
            case["evidence"],
            case["claim"]
        )

        expected = case["expected"]

        expected_labels.append(expected)
        predicted_labels.append(prediction)

        status = "PASS" if prediction == expected else "FAIL"

        print(f"\nTest {index}: {status}")
        print("Evidence:", case["evidence"])
        print("Claim:", case["claim"])
        print("Expected:", expected)
        print("Predicted:", prediction)
        print("Scores:", {
            label: round(score, 3)
            for label, score in scores.items()
        })

    labels = ["entailment", "contradiction", "neutral"]

    correct = sum(
        expected == predicted
        for expected, predicted in zip(
            expected_labels, predicted_labels
        )
    )

    accuracy = correct / len(test_cases)

    print("\n=== EVALUATION REPORT ===")
    print(f"Correct predictions: {correct}/{len(test_cases)}")
    print(f"Accuracy: {accuracy:.2%}")

    print("\nClassification report:")
    print(classification_report(
        expected_labels,
        predicted_labels,
        labels=labels,
        zero_division=0
    ))

    print("Confusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(
        expected_labels,
        predicted_labels,
        labels=labels
    ))


if __name__ == "__main__":
    evaluate_model()
