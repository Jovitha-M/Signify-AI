import spacy

nlp = spacy.load("en_core_web_sm")

REMOVE_WORDS = {"a", "an", "the", "is", "are", "was", "were", "am", "be"}

TIME_MARKERS = {"yesterday", "today", "tomorrow", "morning", "afternoon", "night", "week", "month", "year"}
def is_time_marker(word):
    return word.lower() in TIME_MARKERS

def english_to_asl_gloss(sentence):
    doc = nlp(sentence) 
    asl_gloss = []
    time_part = []
    wh_question = None
    verb = None 
    
    for token in doc:
        if token.text.lower() in REMOVE_WORDS:  # Remove unnecessary words
            continue
        
        if is_time_marker(token.text):  # Move time markers to the front
            time_part.append(token.text.upper())
            continue

        if token.dep_ == "ROOT":  # Capture main verb
            verb = token.text.upper()
            continue
        
        if token.dep_ in {"nsubj", "nsubjpass"} and token.text.lower() not in REMOVE_WORDS:  # Subject handling
            asl_gloss.append(token.text.upper())
            continue
        
        if token.dep_ == "dobj":  # Object handling
            asl_gloss.append(token.text.upper())
            continue
        
        if token.dep_ == "advmod" and token.text.lower() in {"where", "why", "how", "what", "who"}:  # WH-questions
            wh_question = token.text.upper()
            continue
        
        asl_gloss.append(token.text.upper())  # Default case for remaining words
    
    # Ensure the verb is included
    if verb:
        asl_gloss.append(verb)
    
    # Construct final sentence in ASL order
    final_gloss = " ".join(time_part + asl_gloss)
    if wh_question:
        final_gloss += f" {wh_question}"  # WH-question at the end
    
    return final_gloss

# Test cases
sentences = [
    "I am going to the store tomorrow.",
    "She is very tired.",
    "Where are you going",
    "If it rains, the game will be canceled.",
    "I give the book to you.",
    "Why are you late",
    "The cat is on the table.",
    "I am learning sign language.",
    "He will go to school next week.",
    "They are my best friends.",
    "Are you hungry",
    "I will meet you at the park.",
    "What do you want to eat",
    "She gave me a gift.",
    "If you study, you will pass.",
    "John loves Mary.",
    "We went to the zoo last Saturday.",
    "Can you help me with this",
    "My brother is a doctor.",
    "She likes to read books."
]

for s in sentences:
    print(f"English: {s}")
    print(f"ASL Gloss: {english_to_asl_gloss(s)}\n")
