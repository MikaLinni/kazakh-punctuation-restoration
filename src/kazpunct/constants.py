LABELS = ("O", "COMMA", "PERIOD", "QUESTION")
LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}
NUM_LABELS = len(LABELS)
IGNORE_INDEX = -100

QUESTION_PARTICLES = frozenset({
    "ма", "ме", "ба", "бе", "па", "пе",
    "мы", "мі", "бы", "бі", "пы", "пі",
    "ше", "шы", "ші",
})

QUESTION_WORDS = frozenset({
    "кім", "не", "қайда", "қашан", "қалай", "неше", "қанша",
    "неліктен", "қайсы", "кімнің", "нені", "кімді", "қандай",
    "қай", "неге", "кімге", "немен", "кімнен", "қайтіп", "қалайша",
})
