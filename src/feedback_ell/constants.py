"""Project-wide constants."""

COMPETITION_SLUG = "feedback-prize-english-language-learning"

TEXT_COLUMN = "full_text"
ID_COLUMN = "text_id"

TARGET_COLUMNS = [
    "cohesion",
    "syntax",
    "vocabulary",
    "phraseology",
    "grammar",
    "conventions",
]

REQUIRED_TRAIN_COLUMNS = [ID_COLUMN, TEXT_COLUMN, *TARGET_COLUMNS]
REQUIRED_TEST_COLUMNS = [ID_COLUMN, TEXT_COLUMN]

SCORE_MIN = 1.0
SCORE_MAX = 5.0
