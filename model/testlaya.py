import os

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

import laya
from laya import Router


def format_answers(answers):
    """Convert Laya's nested prediction data into readable sentences."""
    lines = []

    for name, answer in answers.items():
        answer_type = answer.get("type")
        label = name.replace("_", " ").capitalize()

        if answer_type == "choice":
            value = answer.get("choice", "unknown")
            confidence = answer.get("confidence")
            confidence_text = (
                f" ({confidence:.0%} confidence)" if confidence is not None else ""
            )
            lines.append(f"{label}: {value}{confidence_text}.")
        elif answer_type == "score":
            probabilities = answer.get("probabilities", {})
            legend = answer.get("legend", {})
            most_likely = max(probabilities, key=probabilities.get, default=None)
            urgency = legend.get(str(most_likely), "unknown")
            lines.append(f"{label}: {urgency}.")
        elif answer_type == "noul":
            value = answer.get("noul", 0)
            response = "Yes" if value >= 0.5 else "No"
            lines.append(f"{label}: {response}.")
        else:
            lines.append(f"{label}: {answer}.")

    return "\n".join(lines)


router = Router(preload=True)  # loads all 3 checkpoints into memory

state = "I was charged twice for the same subscription ugh! Please reverse the duplicate charge  or I will have to escalate this issue."

questions = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this?",
        "criteria": ["billing", "technical", "sales"],
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this?",
        "criteria": ["not urgent", "soon", "critical"],
    },
    "refund_requested": {
        "type": "noul",
        "instructions": "Does the customer ask for money back?",
    },
}

result = router.predict(state, questions)
print("\nCustomer request summary")
print(format_answers(result["answers"]))