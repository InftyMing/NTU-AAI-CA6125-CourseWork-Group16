from __future__ import annotations

from pathlib import Path

import pandas as pd

from feedback_ell.constants import TARGET_COLUMNS


def main() -> None:
    output = Path("data/raw")
    output.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "text_id": "demo_001",
            "full_text": "Learning English is important because it helps students communicate with people from many countries. I practice reading and writing every week, and I try to use new words in my essays.",
            "cohesion": 3.5,
            "syntax": 3.0,
            "vocabulary": 3.5,
            "phraseology": 3.0,
            "grammar": 3.0,
            "conventions": 3.5,
        },
        {
            "text_id": "demo_002",
            "full_text": "The city should build more parks. Parks give children a place to play and they also make the air better. Some people say buildings are more useful, but green spaces are necessary for healthy communities.",
            "cohesion": 4.0,
            "syntax": 4.0,
            "vocabulary": 4.0,
            "phraseology": 3.5,
            "grammar": 4.0,
            "conventions": 4.0,
        },
        {
            "text_id": "demo_003",
            "full_text": "I think school lunch need change because many student no like it. The food sometimes cold and not have much vegetable. If school ask students, lunch can be more good.",
            "cohesion": 2.5,
            "syntax": 2.0,
            "vocabulary": 2.5,
            "phraseology": 2.0,
            "grammar": 2.0,
            "conventions": 2.5,
        },
        {
            "text_id": "demo_004",
            "full_text": "Technology can improve education when teachers use it carefully. Online resources provide examples, practice, and feedback, but students still need discussion and guidance from teachers.",
            "cohesion": 4.0,
            "syntax": 4.5,
            "vocabulary": 4.0,
            "phraseology": 4.0,
            "grammar": 4.5,
            "conventions": 4.0,
        },
        {
            "text_id": "demo_005",
            "full_text": "My favorite season is summer because I can visit my grandparents and swim with my cousins. The weather is hot, but the long holiday gives me time to read books and relax.",
            "cohesion": 3.5,
            "syntax": 3.5,
            "vocabulary": 3.0,
            "phraseology": 3.5,
            "grammar": 3.5,
            "conventions": 3.5,
        },
        {
            "text_id": "demo_006",
            "full_text": "Uniforms are useful in school because they reduce pressure about clothing. However, students should still have some freedom, such as choosing comfortable shoes or jackets.",
            "cohesion": 4.0,
            "syntax": 4.0,
            "vocabulary": 3.5,
            "phraseology": 4.0,
            "grammar": 4.0,
            "conventions": 4.0,
        },
    ]
    train = pd.DataFrame(rows)
    test = train[["text_id", "full_text"]].head(3).copy()
    test["text_id"] = ["demo_test_001", "demo_test_002", "demo_test_003"]
    sample = test[["text_id"]].copy()
    for col in TARGET_COLUMNS:
        sample[col] = 0.0
    train.to_csv(output / "train.csv", index=False)
    test.to_csv(output / "test.csv", index=False)
    sample.to_csv(output / "sample_submission.csv", index=False)
    print("Demo data written to data/raw.")


if __name__ == "__main__":
    main()
