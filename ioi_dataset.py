NAMES = [
    ("John", "Mary"),
    ("Alice", "Bob"),
    ("Tom", "Sarah"),
    ("James", "Emma"),
    ("Luke", "Anna"),
]

def make_ioi_example(a, b):
    clean = (
        f"When {a} and {b} went to the store, "
        f"{a} gave a book to"
    )

    corrupt = (
        f"When {a} and {b} went to the store, "
        f"{b} gave a book to"
    )

    return {
        "clean": clean,
        "corrupt": corrupt,
        "correct": b,
        "incorrect": a,
    }

DATASET = [
    make_ioi_example(a, b)
    for a, b in NAMES
]