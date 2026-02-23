"""
Built-in sample test cases with expected hallucination types
re-evaluated against the specification document definitions.

Expected type reasoning:
────────────────────────────────────────────────────────────
Test 1 → TYPE 1A
  Answer says "Steven Spielberg" directed Inception.
  Spielberg does NOT appear in the paragraph at all.
  The answer introduces a new, ungrounded entity → 1A.

Test 2 → TYPE 1B
  Answer says Apple was founded by "Bill Gates, Steve Jobs, Paul Allen".
  Bill Gates and Paul Allen are real entities (Microsoft founders) but they
  are incorrectly PAIRED with Apple. Wrong entity–entity association → 1B.

Test 3 → TYPE 1A
  The paragraph only mentions Martin Eberhard and Marc Tarpenning as founders.
  The answer introduces "Elon Musk" — an entity NOT present in or inferable
  from the paragraph. New ungrounded entity added → 1A.
  (Note: the date "2003" is correct, but the Elon Musk claim is the hallucination.)

Test 4 → TYPE 2A
  The entities are correct (the novel, surveillance). But the INTENT is inverted:
  the paragraph frames surveillance as oppressive government control, while the
  answer says it "promotes stability and social harmony". Predicate distortion → 2A.

Test 5 → TYPE 1B
  No paragraph → world knowledge is ground truth.
  "Brazil" is a real entity; "2018 FIFA World Cup winner" is a real attribute.
  But they are incorrectly PAIRED — France won, not Brazil.
  Wrong entity–attribute combination → 1B.
  (The answer's second sentence correctly states France won — classify on the
   hallucinated claim only, per Rule 5.)
────────────────────────────────────────────────────────────
"""

SAMPLE_TESTS = [
    {
        # Expected: TYPE 1A — "Steven Spielberg" not in paragraph (grounding violation)
        "paragraph": (
            "Inception is a 2010 science fiction film directed by Christopher Nolan. "
            "The film stars Leonardo DiCaprio as Dom Cobb, a thief who steals information "
            "by infiltrating dreams. The movie explores themes of reality, memory, and "
            "subconscious manipulation."
        ),
        "question": "Who directed Inception?",
        "answer": "Inception was directed by Steven Spielberg.",
    },
    {
        # Expected: TYPE 1B — Bill Gates & Paul Allen are real but incorrectly paired with Apple
        "paragraph": (
            "Apple Inc. was founded in 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne. "
            "The company is known for products such as the iPhone, iPad, and Mac computers."
        ),
        "question": "Who were the founders of Apple?",
        "answer": "Apple was founded by Bill Gates, Steve Jobs, and Paul Allen.",
    },
    {
        # Expected: TYPE 1A — "Elon Musk" not present in / inferable from the paragraph
        "paragraph": (
            "Tesla, Inc. was founded in 2003 by engineers Martin Eberhard and Marc Tarpenning. "
            "The company specializes in electric vehicles and renewable energy solutions."
        ),
        "question": "When was Tesla founded?",
        "answer": (
            "Tesla was founded in 2003 and is currently led by Elon Musk, "
            "who transformed it into the world's most valuable car company."
        ),
    },
    {
        # Expected: TYPE 2A — entities correct, intent/predicate inverted
        "paragraph": (
            "Nineteen Eighty-Four is a novel by George Orwell that depicts a totalitarian "
            "society under constant surveillance. The story explores themes of government "
            "control, truth manipulation, and loss of individual freedom."
        ),
        "question": "What is the central theme of 1984?",
        "answer": (
            "The novel primarily promotes the idea that strong surveillance systems "
            "create stability and social harmony."
        ),
    },
    {
        # Expected: TYPE 1B — Brazil (real entity) incorrectly paired with "2018 World Cup winner"
        # No paragraph → world knowledge used. Second sentence in answer is correct;
        # classification is based on the wrong first sentence only.
        "paragraph": "",
        "question": "Who won the 2018 FIFA World Cup?",
        "answer": (
            "The winner of the 2018 FIFA World Cup was Brazil national football team.\n"
            "The 2018 FIFA World Cup was held in Russia. In the final match, "
            "France national football team defeated Croatia national football team 4-2 "
            "to win the tournament."
        ),
    },
]
