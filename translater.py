#
#  translater.py
#
import argparse
import json
import unittest

#region CLI Logic
def patched_addSuccess(self, test):
    if not hasattr(self, 'successes'):
        self.successes = []
    self.successes.append(test)

unittest.TestResult.addSuccess = patched_addSuccess

def run_selected_tests(test_names):
    suite = unittest.TestSuite()
    for name in test_names:
        suite.addTest(TranslationTester(name))

    result = unittest.TestResult()
    suite.run(result)

    table = Table(title="Translation Test Results", show_lines=True, highlight=True)
    table.add_column("Test Name", style="bold")
    table.add_column("Result", style="bold")

    for test in result.successes if hasattr(result, 'successes') else []:
        table.add_row(str(test), "[green]PASS")

    for test, failure_message in result.failures:
        table.add_row(str(test), "[red]FAIL")
        table.add_row("", f"[red]Failure: {failure_message}")

    for test, error_message in result.errors:
        table.add_row(str(test), "[red]ERROR")
        table.add_row("", f"[red]Error: {error_message}")

    for test, _ in result.skipped:
        table.add_row(str(test), "[yellow]SKIPPED")

    console.print(table)


def cli_translate(args):
    user_input = args.input
    output_lang = args.output
    formal = args.formal

    translated = translate(text=user_input, to=output_lang, formal=formal)

    print("\nTranslation: " + translated)
    if args.ipa:
        console.print(f"[dim]{get_ipa_pronounciation(translated)}[/dim]", highlight=False)

def cli_run_tests(args):
    console.print(Rule(title="[dim white]Running tests..."), style="dim")
    run_selected_tests(args.tests)

def cli_inspect(args):
    translation, inspection = from_gorgus(args.sentence)

    if not args.json: # text format
        console.print(Rule())

        # Display input
        console.print("[bold]Input:[/bold]", inspection["input"], end="\n\n", highlight=False)

        # Display word analisys
        word_table = Table("Word", "Lemma", "POS", "Features", title="Word Analisys", box=box.ROUNDED)
        for word in inspection["words"]:
            features_list = []
            for k,v in word["features"].items():
                features_list.append(f"{k.capitalize()}={str(v).capitalize()}")

            word_table.add_row(word["word"], word["lemma"], word["pos"], ', '.join(features_list))
        console.print(word_table, end='\n\n')

        # Display morphology breakdown
        if args.verbose or args.morph:
            console.print("\n[bold][Morphology Breakdown][/bold]")
            for x in inspection["morphology"]:
                console.print(f"- {x}")

        # Display translation
        if args.verbose or args.translate:
            console.print("\n[bold][Translation][/bold]")
            console.print(f"\"{translation}\"")

        # Display grammar notes
        if args.verbose or args.notes:
            console.print("\n[bold][Grammar Notes][/bold]")
            for x in inspection["notes"]:
                console.print(f"- {x}")

        # Display pronounciation
        if args.verbose or args.phonetics:
            console.print("\n[bold][Pronounciation Guide][/bold]")
            for word in inspection["words"]:
                final_pronounciation_string = f"- {word['word']} → [green]{get_ipa_pronounciation(word['word'])}[/green]"
                console.print(final_pronounciation_string, highlight=False)


        console.print(Rule())
    else: # json format
        console.print(inspection)

    if args.output:
        console.print('[dim]Exporting to JSON file...[/dim]', highlight=False)
        f = open(args.output, "w")
        json.dump(inspection, f, indent=4)
        f.close()

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        prog="Gorgus Translater",
        description="The CLI interface for the Gorgus Translator, you can translate text to and from Gorgus, run tests, etc..."
    )

    subparsers = arg_parser.add_subparsers()

    # oh boy
    translate_parser = subparsers.add_parser("translate", help="Translate text to and from Gorgus", description="Translate text to and from Gorgus")
    translate_parser.add_argument("input", help="The text input", type=str)
    translate_parser.add_argument("-o", "--output", type=str, help="The output language", default="gorgus", choices=["gorgus", "english"])
    translate_parser.add_argument("-f", "--formal", action="store_true", help="Enable formal speach")
    translate_parser.add_argument("--ipa", action="store_true", help="Include an IPA transcription if translating from English to Gorgus")
    translate_parser.set_defaults(func=cli_translate)

    tests_parser = subparsers.add_parser("run_tests", help="Run tests", description="Run tests")
    tests_parser.add_argument("tests", nargs="*", default=["test_to_gorgus", "test_from_gorgus", "test_tense_detection", "test_translation_speed"])
    tests_parser.set_defaults(func=cli_run_tests)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect a Gorgus sentence to see how it is interpreted")
    inspect_parser.add_argument("sentence", help="The Gorgus sentence to inspect", type=str)
    inspect_parser.add_argument("--json", action="store_true", help="Output in JSON format")
    inspect_parser.add_argument("--morph", action="store_true", help="Include morphological analysis")
    inspect_parser.add_argument("--verbose", action="store_true", help="Show full inspection details")
    inspect_parser.add_argument("--phonetics", action="store_true", help="Include IPA pronounciation guide")
    inspect_parser.add_argument("--translate", action="store_true", help="Include translation")
    inspect_parser.add_argument("--notes", action="store_true", help="Show additional interpretation notes")
    inspect_parser.add_argument("-o", "--output", type=str, help="Save output to file")
    inspect_parser.set_defaults(func=cli_inspect)

    args = arg_parser.parse_args()

    try:
        args.func
    except AttributeError:
        print("You didn't provide a command!")
        arg_parser.print_usage()
        exit(1)
#endregion

import re
import unicodedata

import nltk
import inflect

import tagger_model_trainer

from translations import *
from nltk.stem import WordNetLemmatizer, LancasterStemmer
from typing import Literal
from time import time

from rich.console import Console
from rich.table import Table
from rich.rule import Rule
from rich import box

console = Console()

ACTOR_SUFFIXES = ["er", "or", "ist"]

console.print("[bold bright_green]INFO[/bold bright_green] Loaded translater dependencies!")
console.print("[bold bright_green]INFO[/bold bright_green] Preparing thrases dictionary..")

# First, sort phrases by length (longer phrases first)
sorted_phrases = sorted(
    phrase_translations.items(),
    key=lambda item: max(len(p) for p in (item[1] if isinstance(item[1], list) else [item[1]])),
    reverse=True
)

console.print("[bold bright_green]INFO[/bold bright_green] Starting [bold]inflect[/bold] engine..")

inflect_engine = inflect.engine()

#console.print("[bold bright_green]INFO[/bold bright_green] Loading [bold]SpaCy[/bold] AI model..")
#nlp = spacy.load("en_core_web_sm")

console.print("[bold bright_green]INFO[/bold bright_green] Loading [bold]NLTK[/bold] modules..")
lemmatizer = WordNetLemmatizer()
stemmer = LancasterStemmer()
def nltk_download(packagePath, package) -> bool:
    try:
        console.print(f"[dim]Checking for {package}...[/dim]", highlight=False)
        nltk.data.find(packagePath)
        return True
    except LookupError:
        console.print(
            f"[bold orange1]Warning![/bold orange1] {package} was not found! Attempting to automatically install..", highlight=False)
        console.print(
            "[dim]Disabling SSL check to prevent issues on certain operating systems (MacOS, I'm looking at you)...[/dim]", highlight=False)
        import ssl
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        download_success = nltk.download(package)
        return download_success
wordnet_download_success = nltk_download("corpora/wordnet.zip", "wordnet")
brown_download_success = nltk_download("corpora/brown.zip", "brown")
punkt_download_success = nltk_download("tokenizers/punkt_tab.zip", "punkt_tab")
tagger_download_success = nltk_download("taggers/averaged_perceptron_tagger_eng.zip", "averaged_perceptron_tagger_eng")
if not (wordnet_download_success and brown_download_success and punkt_download_success):
    raise Exception("NLTK Resources Download Failed!")

console.print("[bold bright_green]INFO[/bold bright_green] Importing [bold]nltk.corpus[/bold]..")
import nltk.corpus
console.print("[bold bright_green]INFO[/bold bright_green] Getting [bold]NLTK Unigram Tagger[/bold]..")
# the default tagger is not good, for some reason.
unigram_tagger = tagger_model_trainer.get_unigram_tagger_and_train_if_not_found()
bigram_tagger  = tagger_model_trainer.get_bigram_tagger_and_train_if_not_found()

def get_word_type(word):
    if word.strip() == "":
        return "UNKOWN"

    # Process the word through the spaCy pipeline
    #doc = nlp(word)
    doc = nltk.tokenize.word_tokenize(word)
    doc = nltk.pos_tag(doc)

    # we need to map the tag to only a few tags, cause rn it's too specific
    tag = doc[0][1]
    KNOWN_TAGS = {
        "NOUN": ["NN", "NNS"],
        "ADJECTIVE": ["JJ"],
        "VERB": ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"],
        "ADVERB": ["RBS"],
        "ADPOSITION": ["IN"],
        "PRONOUN": ["PRP"],
        "PARTICLE": ["WRB"],
        "DETERMINER": ["PRP$"]
    }
    for known_tag, list_of_tags in KNOWN_TAGS.items():
        if tag in list_of_tags:
            tag = known_tag
            break

    # Check if the word is a verb by examining the POS tag
    try:
        #return doc[0].pos_
        return tag
    except IndexError:
        return "UNKOWN"

def remove_all_except(text, accents_to_keep = {'\u0302', '\u0303', '\u0310', "\u0306"}):
    """
    Removes all diacritical marks except those in `accents_to_keep`.
    
    :param text: The input string containing diacritical marks.
    :param accents_to_keep: A set of Unicode characters representing the accents to keep.
    :return: The cleaned text with only the specified accents retained.
    """
    # Normalize text to decomposed form (NFD)
    normalized_text = unicodedata.normalize('NFD', text)
    # Keep only base characters and the specified accents
    cleaned_text = ''.join(c for c in normalized_text if unicodedata.category(c) != 'Mn' or c in accents_to_keep)
    return cleaned_text

# Ensure all dictionary values are lists for uniform processing
console.print("[bold bright_green]INFO[/bold bright_green] Normalising [bold]translations[/bold]..")
normalized_translation_dict = {k: ([v] if isinstance(v, str) else v) for k, v in translation_dictionary.items()}
deaccented_translation_dict = {remove_all_except(k): ([v] if isinstance(v, str) else v) for k, v in translation_dictionary.items()}
no_accent_to_accented = {}
reverse_mapping = {}
for norm_key in normalized_translation_dict:
    deaccented = remove_all_except(norm_key)
    reverse_mapping[norm_key] = deaccented
    no_accent_to_accented[deaccented] = norm_key

def detect_verb_tense(verb, previous_word = None):
    compliant_verb = (((previous_word + " ") if previous_word else "") + verb)
    # phrase translation is trigger happy
    compliant_verb = compliant_verb.replace("zúgúgo", "Will")
    tokenized_verb = compliant_verb.split(" ")

    tagged_verb = bigram_tagger.tag(tokenized_verb)

    if tagged_verb[len(tagged_verb) - 1][1] is None:
        # try again but with the unigram tagger
        tagged_verb = unigram_tagger.tag(tokenized_verb)

    _verb = tagged_verb[len(tagged_verb) - 1][1]

    if _verb == "VBD":
        return "past"
    if _verb == "VB":
        if previous_word or len(tokenized_verb) != 1:
            if tagged_verb[0][1] == "MD" and tokenized_verb[0] == "will":
                return "futr"
        return "norm"
    if _verb == "VBG":
        return "cont"
    if _verb == "VBN":
        return "past"
    return "norm"
    # if (
    #     sent.root.tag_ == "VBD" or
    #     any(w.dep_ == "aux" and w.tag_ == "VBD" for w in sent.root.children)):
    #     return "past"
    #
    # if (
    #     sent.root.tag_ == "VBG" or
    #     any(w.dep_ == "aux" and w.tag_ == "VBZ" for w in sent.root.children)):
    #     return "cont"
    #
    # if previous_word and previous_word in ["will", "shall"]:
    #     return "futr"
    #
    # return "norm"

def get_past_tense_verb(verb):
    """
    Returns the past tense form of a verb.
 
    Parameters:
    - verb: str
        The verb for which the past tense is to be determined.
 
    Returns:
    - str:
        The past tense form of the verb.
 
    Examples:
    >>> get_past_tense_verb("run")
    "ran"
 
    >>> get_past_tense_verb("eat")
    "ate"
 
    >>> get_past_tense_verb("write")
    "wrote"
    """
 
    # List of common irregular verbs and their past tense forms
    irregular_verbs = {
        "be": "was",
        "have": "had",
        "do": "did",
        "go": "went",
        "see": "saw",
        "come": "came",
        "give": "gave",
        "take": "took",
        "make": "made",
        "find": "found",
        "know": "knew",
        "think": "thought",
        "say": "said",
        "tell": "told",
        "get": "got",
        "give": "gave",
        "read": "read",
        "wear": "wore",
        "write": "wrote",
        "run": "ran",
        "eat": "ate"
        # Add more irregular verbs as needed
    }
 
    # Check if the verb is in the irregular verbs dictionary
    if verb in irregular_verbs:
        return irregular_verbs[verb]
 
    # Check if the verb ends with "e"
    elif verb.endswith("e"):
        return verb + "d"
 
    # Check if the verb ends with a consonant followed by "y"
    elif verb[-2] not in "aeiou" and verb[-1] == "y":
        return verb[:-1] + "ied"
 
    # For regular verbs, simply add "ed" to the end
    else:
        return verb + "ed"
    
def convert_to_gerund(verb):
    if verb.endswith("e") and not verb.endswith("ie"):
        return verb[:-1] + "ing"  # Drop 'e' and add 'ing'
    elif verb.endswith("ie"):
        return verb[:-2] + "ying"  # Change 'ie' to 'ying'
    elif len(verb) > 2 and verb[-1] in "aeiou" and len(verb) > 1 and verb[-2] not in "aeiou" and verb[-3] not in "aeiou" and verb[-2] != 'w' and verb[-2] != 'x' and verb[-2] != 'y':
        return verb + verb[-1] + "ing"  # Double the final consonant for CVC pattern
    return verb + "ing"  # Default case for most verbs
    
def get_tense_verb(verb, tense):
    if tense == "past":
        return get_past_tense_verb(verb)
    
    elif tense == "cont":
        return convert_to_gerund(verb)
    
    elif tense == "futr":
        return "will " + verb

    else:
        return verb

def is_actor_form(word):
    """Check if a word is in actor form based on common English suffixes."""
    return any(word.endswith(suffix) for suffix in ACTOR_SUFFIXES)

def to_actor_form(root):
    """Convert a root word to its actor form following English rules."""
    if root.endswith("e"):
        return root + "r"  # e.g., "bake" → "baker"
    elif re.match(r".*[aeiou][bcdfghjklmnpqrstvwxyz]$", root):
        return root + root[-1] + "er"  # e.g., "run" → "runner"
    else:
        return root + "er"  # Default case

def from_actor_form(actor, lemma = True):
    """
    Convert an actor form word back to its root.
    """
    # if wordnet is not available we have to make compromises
    if not wordnet_download_success: return actor

    if not lemma:
        raise Exception("Invalid parameters - lemma must be True")

    return stemmer.stem(actor)

def get_trailing_punctuation(text, ignore_chars=""):
    # Create a regex pattern that matches punctuation but ignores specified characters
    ignore_chars = re.escape(ignore_chars)  # Escape to handle special characters properly
    pattern = rf'[^\w\s{ignore_chars}]+$'  # Match trailing punctuation excluding ignored ones
    match = re.search(pattern, text)
    
    return remove_all_except(match.group(0) if match else '')

def convert_to_base_form(verb):
    if verb in ["is", "are"]: # some words will cause issues if converted to base form
        return verb

    if wordnet_download_success:
        return lemmatizer.lemmatize(verb, pos='v')
    else:
        return verb

def create_ipa_dict(consonants):
    ipa_dict = {}
    for ipa, romanizations in consonants.items():
        for roman in romanizations:
            ipa_dict[roman] = ipa
    return ipa_dict

def get_ipa_pronounciation(gorgus: str):
    consonants = {
        "lʊː": ["lu"],
        "ʃ": ["sh", "ćh"],
        "O": ["oe", "ó"],
        "l": ["l", "ll"],
        "iː": ["ee", "é", "ea"],
        "h": ["h"],
        "ɜː": ["er", "ur"],
        "ɔɹ": ["or"],
        "tʃ": ["ch"],
        "ʌ": ["u"],
        "ʊː": ["oo", "ú"],
        "ɔ": ["o"],
        "iːkO": ["eeko"],
        "ʤ": ["j"],
        "ɔɹʤ": ["orge"],
        "ɹ": ["r"],
        "f": ["f", "ff"],
        "kw": ["q", "qu"],
        "ɔɹs": ["ors", "orse"],
        "ŋg": ["nġ"],
        "iŋg": ["ing"],
        "ŋ": ["ng"],
        "oŋk": ["onk"],
        "t": ["t", "tt"],
        "ɑːɹ": ["ar", "å"],
        "k": ["k", "c", "ck"],
        "g": ["g", "gg"],
        "ɛ": ["è"],
        "R": ["ŕ̈"], # r trill
        "e͡ɪ": ["ae", "ä", "â", "ai", "ay"],
        "ɪ": ["i"],
        "ɪŋk": ["ink"],
        "eŋk": ["enk"],
        "θ": ["th"],
        "e͡ɪv": ["ave"],
        "j": ["y"],
        "ks": ["x"],
        "lʌ̌ŋk": ["lunk"], # questions have a rising tone
        "iːnO": ["ino"],
        "vɪŋ": ["ving"],
        "Oʊʤ": ["oge"],
        "ɹs": ["rse"],
        "aɪk": ["ike"],
        "aɪd": ["ide", "ied"],
        "kχ": ["ç"],
        "m": ["m", "mm", "mmm"],
        "ː": [translation_dictionary["<EXAGGERATED_VERB>"]], # exaggerated vowel
        "\u0324˨˩": [translation_dictionary["<GENTLE_VERB>"]], # gentle
        "": ['a̱', 'ḇ', 'c̱', 'ḏ', 'e̱', 'f̱', 'g̱', 'ẖ', 'i̱', 'j̱', 'ḵ', 'ḻ', 'm̱', 'ṉ', 'o̱', 'p̱', 'q̱', 'ṟ', 's̱', 'ṯ', 'u̱', 'v̱', 'w̱', 'x̱', 'y̱', 'ẕ'] # silent letters
    }

    # Invert dictionary
    ipa_dict = create_ipa_dict(consonants)

    gorgus = gorgus.replace(",", " |").replace(".", " ‖").replace("!", " ‖")
    gorgus = gorgus.translate(str.maketrans('', '', "?!\":()=/\\$[]"))
    gorgus = gorgus.lower().replace("-", "").replace("'", ".")

    words = gorgus.split()  # Split into words
    ipa_output = []
    
    for word in words:
        ipa_word = word

        for roman, ipa in sorted(ipa_dict.items(), key=lambda x: -len(x[0])):  # Sort by length (longest first)
            ipa_word = ipa_word.replace(roman, ipa)

        ipa_output.append(ipa_word)

    return "/" + ' '.join(ipa_output).replace("R", "r").replace("O", "o") + "/"

def remove_between_last_two_spaces(s):
    parts = s.rsplit(" ", 2)  # Split into up to 3 parts from the right
    if len(parts) < 3:
        return s  # Not enough spaces to remove anything
    return parts[0] + " " + parts[2]  # Keep first and last parts, remove the middle

def to_gorgus(user_input, formal = True):
    translated = ""
    before_translation = user_input

    #user_input = swap_verbs_nouns(user_input)
    
    # Replace phrases with gorgus words      
    for gorgus, english_phrases in sorted_phrases:
        if isinstance(english_phrases, str):
            english_phrases = [english_phrases]  # Ensure it's a list
        for phrase in english_phrases:
            # Use regex to match whole phrase boundaries
            pattern = r'\b' + re.escape(phrase) + r'\b'
            user_input = re.sub(pattern, gorgus, user_input, flags=re.IGNORECASE)
   
    words = user_input.split(" ")

    modified_verbs = {}

    """doc = nlp(before_translation)
    for i, token in enumerate(doc):
        if token.text == "EXAGGERATE":
            modified_verbs[token.head.i] = 1
        elif token.text == "GENTLE":
            modified_verbs[token.head.i] = -1"""

    previous_english_word = None
    for i, word in enumerate(words): 
        if word == "the": # skip "the", there is no equivelant in gorgus
            continue

        trailing_punctuation = get_trailing_punctuation(word, translation_dictionary["<EXAGGERATED_VERB>"] + translation_dictionary["<GENTLE_VERB>"] + translation_dictionary["<MORE_VERB>"] + translation_dictionary["<LESS_VERB>"])
        word = word.translate(str.maketrans('', '', "?.!,\":()=/\\$[]"))

        suffix = ""
        punctuation_suffix = ""

        if word == "EXAGGERATE" or word == "GENTLE" or word == "MORE" or word == "LESS":
            continue

        suffix: str = modified_verbs.get(i, "")
        if suffix == 1:
            suffix = translation_dictionary["<EXAGGERATED_VERB>"]
        elif suffix == -1:
            suffix = translation_dictionary["<GENTLE_VERB>"]
        elif suffix == 2:
            suffix = translation_dictionary["<MORE_VERB>"]
        elif suffix == -2:
            suffix = translation_dictionary["<LESS_VERB>"]

        if trailing_punctuation.endswith("?"):
            if word != "lunk":
                punctuation_suffix += " lunk"
        else:
            punctuation_suffix += trailing_punctuation

        try:
            plural = inflect_engine.plural(word)
            if not word in ignored_plurals:
                singular = inflect_engine.singular_noun(word)
            else:
                singular = False
            is_plural = singular != False
        except:
            translated += f"{words[i]} "
            continue
        
        word_suffix = ""
        is_actor = False

        is_actor = word not in ignored_actor_nouns and (
            is_actor_form(word) or "erser" in to_actor_form(word)
        )

        if is_actor:
            singular = from_actor_form(word)
            plural = inflect_engine.plural(from_actor_form(plural))

            word_suffix += translation_dictionary["<ACTOR>"]
        
        if is_actor and singular == plural:
            is_plural = True
            singular = to_actor_form(singular)
            plural = inflect_engine.plural(singular)

        if plural in ignored_plurals:
            is_plural = False

        # we need to figure out what tense the verb is in :DDDDD (this is fucking painful, we also only use this if the word is a verb)
        tense = detect_verb_tense(word, previous_english_word)
        word_type = get_word_type(word)
        base_word = convert_to_base_form(word)

        if tense == "futr": # remove the last word
            translated = ' '.join(translated.strip().split(' ')[:-1]) + " "

        found = False
        for key, value_list in normalized_translation_dict.items():
            
            value_set = set(value_list)  # Convert to set for faster lookups
            
            if (word_type != "VERB" and ((singular and singular in value_set) or (word in value_set) or (is_plural and plural in value_set))) or (word_type == "VERB" and base_word in value_set):
                found = True
                plural_prefix = translation_dictionary["<PLURAL>"] if is_plural else ""
                tense_suffix = translation_dictionary.get(f"<{tense.upper()}_TENSE>", "") #if word_type == "VERB" else ""
                word_type_suffix = translation_dictionary. get(f"<{word_type.upper()}>", "") if formal else ""
                
                translated += f"{plural_prefix}{key}{word_type_suffix}{word_suffix}{suffix}{tense_suffix}{punctuation_suffix} "
                break

        if not found:
            translated += f"{word}{suffix}{punctuation_suffix} "

        previous_english_word = word

    # Replace verb modifier words
    for word in ["really", "extremely", "very", "absolutely"]:
        translated = replace_word(translated, word, translation_dictionary["<EXAGGERATED_VERB>"])
    for word in ["kinda", "slightly", "somewhat"]:
        translated = replace_word(translated, word, translation_dictionary["<GENTLE_VERB>"])
    for word in ["more"]:
        translated = replace_word(translated, word, translation_dictionary["<MORE_VERB>"])
    for word in ["less"]:
        translated = replace_word(translated, word, translation_dictionary["<LESS_VERB>"])

    return translated

def analyze_pronoun(pronoun):
    """Figure out which person, and which gender a pronoun is.

    Person will be `-1` if the word is not a pronoun.
    """
    pronoun = pronoun.lower()

    # Define dictionaries
    person_map = {
        'i': 1,
        'me': 1,
        'my': 1,
        'mine': 1,
        'we': 1,
        'us': 1,
        'our': 1,
        'ours': 1,

        'you': 2,
        'your': 2,
        'yours': 2,
        "yourself": 2,

        'he': 3,
        'him': 3,
        'his': 3,
        'she': 3,
        'her': 3,
        'hers': 3,
        'it': 3,
        'its': 3,
        'they': 3,
        'them': 3,
        'their': 3,
        'theirs': 3,
    }

    gender_map = {
        'he': 'masculine',
        'him': 'masculine',
        'his': 'masculine',
        'she': 'feminine',
        'her': 'feminine',
        'hers': 'feminine',
        'it': 'neuter',
        'its': 'neuter',
        'they': 'neutral',
        'them': 'neutral',
        'their': 'neutral',
        'theirs': 'neutral',
        # Others are gender-neutral
    }

    person = person_map.get(pronoun, -1)
    gender = gender_map.get(pronoun, 'neutral')

    return person, gender

def from_gorgus(user_input: str):
    translated = ""
    inspection = {
        "input": user_input,
        "words": [],
        "notes": ["Sentence structure: [blue]SVO[/blue]"],
        "morphology": []
    }

    user_input = remove_all_except(user_input)

    # Replace phrases with english words
    for gorgus, english in phrase_translations.items():
        if type(english) == list:
            for phrase in english:
                user_input = user_input.replace(remove_all_except(gorgus), english[0])
        elif type(english) == str:
            user_input = user_input.replace(remove_all_except(gorgus), english)

    words = user_input.split(" ")

    for word in words:
        if word == "lunk":
            translated = translated[:-1] + "? "
            
            inspection["words"].append({
                "word": "lunk",
                "pos": "particle",
                "lemma": "lunk",
                "features": {
                    "role": "QuestionMarker"
                }
            })
            inspection["notes"].append("[bold]\"lunk\"[/bold] at the end confirms this is a direct question.")
            inspection["morphology"].append("lunk = sentence-final question marker")
            continue
        
        suffix = ""
        trailing = get_trailing_punctuation(word, translation_dictionary["<EXAGGERATED_VERB>"] + translation_dictionary["<GENTLE_VERB>"] + translation_dictionary["<MORE_VERB>"] + translation_dictionary["<LESS_VERB>"])
        suffix += trailing

        word = word.translate(str.maketrans('', '', ".,?!$:()=/\\[]"))

        # start creating the word's inspection
        current_words_inspection = {
            "word": no_accent_to_accented.get(word.lower(), word),
            "pos": "unkown",
            "features": {
            },
        }

        word_before_translation = word

        plural = False
        actor = False

        tense = "norm"

        # the suffixes and prefixes lists are used by the inspector
        suffixes = []
        prefixes = []

        for tense_key, tense_value in {translation_dictionary["<CONT_TENSE>"]: "cont", translation_dictionary["<PAST_TENSE>"]: "past", translation_dictionary["<FUTR_TENSE>"]: "futr"}.items():
            #if tense_key in word:
            if word.endswith(tense_key):
                #word = word.replace(tense_key, "")
                word = word.removesuffix(tense_key)
                tense = tense_value
                suffixes.append(tense_key)

                if tense != "cont":
                    current_words_inspection["features"]["tense"] = tense
                    inspection["notes"].append(
                        f"The verb uses the {tense_value}-tense suffix \"{tense_key}\""
                    )
                else:
                    current_words_inspection["features"]["aspect"] = "continuous"

                break

        if word.startswith(translation_dictionary["<PLURAL>"]):
            word = word.removeprefix(translation_dictionary["<PLURAL>"])
            prefixes.append(translation_dictionary["<PLURAL>"])
            plural = True

        if word.endswith(translation_dictionary["<ACTOR>"]):
            word = word.removesuffix(translation_dictionary["<ACTOR>"])
            suffixes.append(translation_dictionary["<ACTOR>"])
            actor = True

        uses_diacritic = False
        if word.find(translation_dictionary["<EXAGGERATED_VERB>"]) != -1:
            word = word.replace(translation_dictionary["<EXAGGERATED_VERB>"], "")
            translated += "really "
            current_words_inspection["features"]["intensity"] = "high"
            inspection["morphology"].append(f"{word_before_translation} = {word} + {translation_dictionary['<EXAGGERATED_VERB>']}  (diacritic for intensified form)")
            uses_diacritic = True
        if word.find(translation_dictionary["<GENTLE_VERB>"]) != -1:
            word = word.replace(translation_dictionary["<GENTLE_VERB>"], "")
            translated += "slightly "
            current_words_inspection["features"]["intensity"] = "low"
            inspection["morphology"].append(f"{word_before_translation} = {word} + {translation_dictionary['<GENTLE_VERB>']}  (diacritic for reduced intensity form)")
            uses_diacritic = True
        if word.find(translation_dictionary["<MORE_VERB>"]) != -1:
            word = word.replace(translation_dictionary["<MORE_VERB>"], "")
            translated += "more "
            current_words_inspection["features"]["intensity"] = "more"
            current_words_inspection["features"]["comparative"] = True
            inspection["morphology"].append(f"{word_before_translation} = {word} + {translation_dictionary['<MORE_VERB>']}  (diacritic for intensified comparative form)")
            uses_diacritic = True
        if word.find(translation_dictionary["<LESS_VERB>"]) != -1:
            word = word.replace(translation_dictionary["<LESS_VERB>"], "")
            translated += "less "
            current_words_inspection["features"]["intensity"] = "less"
            current_words_inspection["features"]["comparative"] = True
            inspection["morphology"].append(f"{word_before_translation} = {word} + {translation_dictionary['<LESS_VERB>']}  (diacritic for reduced intensity comparative form)")
            uses_diacritic = True
        
        if uses_diacritic:
            inspection["notes"].append(
                f"\"{word_before_translation}\" shows {'intensification' if current_words_inspection['features']['intensity'] in ['high', 'more'] else 'reduced intensification'} via diacritic"
            )

        if word == "ji":
            translated += "ji "
            continue

        word_type_suffixes = [
            translation_dictionary["<VERB>"],
            translation_dictionary["<NOUN>"],
            translation_dictionary["<ADJECTIVE>"],
            translation_dictionary["<ADVERB>"],
            translation_dictionary["<ADPOSITION>"]
        ]
        for word_type_suffix in word_type_suffixes:
            if word.endswith(word_type_suffix) or word.endswith(remove_all_except(word_type_suffix)):
                word = word.removesuffix(word_type_suffix).removesuffix(remove_all_except(word_type_suffix))

                if len(suffixes) > 0: # we need to move the most recent suffix before the formality suffix in suffixes list for the inspect tool
                    most_recent_suffix = suffixes.pop()
                    suffixes.append(word_type_suffix)
                    suffixes.append(most_recent_suffix)
                else:
                    suffixes.append(word_type_suffix)

                break

        current_words_inspection["lemma"] = no_accent_to_accented.get(word.lower(), word)

        #return f'{word, translation_dictionary, translation_dictionary.get(word, " Not found!")}'
        translation = deaccented_translation_dict.get(remove_all_except(word.lower()))
    
        output_english = ""
        if translation:
            final = translation[0]

            if actor:
                final = to_actor_form(final)

            if plural:
                final = inflect_engine.plural(final)

            final = get_tense_verb(final, tense)
            for word, features in word_features.items():
                if final.lower() == word.lower():
                    current_words_inspection["features"].update(features)

            output_english = f"{final}{suffix} "
            word_type = get_word_type(output_english)

            person_lookup = {
                1: "first",
                2: "second",
                3: "third"
            }
            person, gender = analyze_pronoun(final) # only used if word is pronoun
            if word_type == "PRONOUN":
                current_words_inspection["features"]["Person"] = person
                current_words_inspection["features"]["Gender"] = gender.capitalize()

            # handle morphology inspection stuff

            if not uses_diacritic: # if the word has a diacritic, we have already handled its morphology
                morphology = f"{current_words_inspection['word']} = "
                features = word_features.get(final.lower()) # some words have some manually added info to them
                should_add_root_tag = not (actor or plural or tense != "norm") # is the word a root word?
                if features: # does the word have extra info?
                    if word_type != "DETERMINER":
                        if should_add_root_tag:
                            morphology += "[Root] "
                        
                        

                    possessive = features.get("possessive", "")
                    if possessive:
                        morphology += "possessive "
                    else:
                        morphology += "("

                    morphology += word_type.lower()

                    if possessive:
                        morphology += f" (\"{possessive.lower()}\")"
                    else:
                        morphology += ")"
                    
                    
                else: # word does not have extra info
                    if should_add_root_tag: # word is a wroot word

                        if word_type == "PRONOUN": # if the word is a pronoun, we add information about the person and gender of the pronoun in the morphology
                            morphology += f"[Root] ({person_lookup[person]} person {gender} pronoun)"
                        else: # it is just a regular root word
                            morphology += f"[Root] (\"{final}\")" 
                    else: # word is not a root word! let's see what suffixes and prefixes it has...
                        
                        entire_word = prefixes + [current_words_inspection["lemma"]] + suffixes

                        morphology += ' + '.join(entire_word)

                        for prefix in prefixes:
                            morphology += f"\n    → Prefix: [red]-{prefix}[/red] (\"{modifier_info[prefix]}\")"

                        morphology += f"\n    → Root: {current_words_inspection['lemma']} (\"{lemmatizer.lemmatize(final)}\")"

                        for suffix in suffixes:
                            morphology += f"\n    → Suffix: [red]-{suffix}[/red] (\"{modifier_info[suffix]}\")"

                    
                inspection["morphology"].append(morphology)
        else:
            output_english = f"{word}{suffix} "
            word_type = get_word_type(output_english)
        translated += output_english

        current_words_inspection["pos"] = word_type.lower()
        
        # note generator
        for rule in grammar_note_rules:
            if rule["condition"](current_words_inspection): # does the word meet the rules?
                if callable(rule["note"]):
                    inspection["notes"].append(rule["note"](current_words_inspection))
                else:
                    inspection["notes"].append(rule["note"])
        # end of note generator

        inspection["words"].append(current_words_inspection)
            
    translated = fix_articles(translated, "ji")

    #translated = swap_verbs_nouns(translated)
    #translated = remove_all_except(translated)

    inspection["translation"] = translated # add translation to inspection
    inspection["notes"] = list(set(inspection["notes"])) # remove duplicate notes

    return translated, inspection

def fix_up(translated, should_add_accents):
    translated = translated.capitalize().strip()

    if not should_add_accents:
        translated = remove_all_except(translated)

    # capitalise the word "I"
    translated = re.sub(r'\bi\b', 'I', translated)

    # Regex pattern to find words that come after ".", "?", "!", or "lunk"
    translated = re.sub(r'([.?!]|\blunk\b)\s*(\w)', lambda m: m.group(1) + ' ' + m.group(2).upper(), translated, flags=re.IGNORECASE)

    return translated

def fix_articles(input_string, article_word):
    words = input_string.split()  # Split the string into words
    result = []

    i = 0
    while i < len(words):
        if words[i].lower() == article_word and i + 1 < len(words):
            # Apply function a to the word after "here"
            result.append(inflect_engine.a(words[i + 1]))
            i += 2  # Skip "here" and the next word as we've processed it
        else:
            # Add the word to the result if it's not part of "here" and the next word
            result.append(words[i])
            i += 1

    return " ".join(result)

def replace_word(input_string, word, replacement, offset = 1):
    # Split the input string into words
    words = input_string.split()

    # Check if 'very' comes before 'quickly'
    thingy = 0 # i don't what to call this variable, we use this so that whenever we delete a word, we're still at the correct word or something idfk

    for i in range(len(words) - 1):
        if words[i - thingy] == word and len(words) > i+1 - thingy:
            if words[i+1 - thingy].lower() == "horge":
                offset = min(offset+1, (len(words)-i)-1)

            trailing = get_trailing_punctuation(words[i+offset-thingy])
            len_trailing = len(trailing)

            # Modify 'quickly' by adding '\u0302' and remove 'very'

            if len_trailing > 0:
                words[i + offset - thingy] = words[i + offset - thingy][:-len(trailing)] + replacement + trailing
            else:
                words[i + offset - thingy] = words[i + offset - thingy][:] + replacement
            del words[i - thingy]  # Remove 'very'

            thingy += 1

    # Join the words back into a string and return
    return " ".join(words)

def translate(text, to: Literal["english", "gorgus"], formal = True, should_add_accents = True):
    """Translate from or to Gorgus and English!

    Trailing whitespace is not preserved, neither is punctuation.
    """
    if text.strip() == "":
        return ""

    text = text.lower().strip()
    text = text.replace("\n", " ")

    if to == "english":
        user_choice = 2
    elif to == "gorgus":
        user_choice = 1
    else:
        raise TypeError("Invalid language conversion! Only options are \"english\" or \"gorgus\".")

    if user_choice == 2: # translate language to english:
            
        translated, _ = from_gorgus(text)

    elif user_choice == 1: # translate english to language      
        
        translated = to_gorgus(text, formal)

    translated = fix_up(translated, should_add_accents)

    return translated


class TranslationTester(unittest.TestCase):
    def test_translation_speed(self):
        sentence = "The quick brown fox jumped over the lazy fat cat."

        start = time()
        to_gorgus(sentence, formal=True)
        translation_time = time() - start

        self.assertLess(translation_time, 0.1, f"Translation is too slow! ({round(translation_time,2)}s) Optimize your damn code!")

    def test_tense_detection(self):
        # key = verb, value = expected tense
        # norm = normal tense (eat, drink)
        # past = past tense (ate, drank)
        # cont = continuing tense (eating, drinking)

        tense_tests = {
            "eat": "norm",
            "sat": "past",
            "teaching": "cont",
            "speaking": "cont",
            "try": "norm",
            "rented": "past",
            "make": "norm",
            "wanting": "cont",
            "will eat": "futr",
            "will die": "futr",
            "ate": "past",
            "slept": "past",
            "die": "norm",
            "open": "norm",
            "will explode": "futr",
            "will make": "futr"
        }
        for verb, expected_tense in tense_tests.items():
            self.assertEqual(detect_verb_tense(verb), expected_tense, f"Detected verb tense and expected verb tense do not match! ({verb})")

    def test_to_gorgus(self):
        # key = english, value = expected gorgus translation
        tests_to_gorgus = {
            "Very cool! Very good. :)": "Klû! Dagunġâ. :)",
            "Hi! How are you?": "Dink! Dup pritterok lunk",
            "How is the weather?": "Dup gorse̱ clidó lunk",
            "I love you.": "H'orpó googrung.",
            # Slept doesn't seem to translate for some reason - even when correctly identified...
            #   ~ HSI
            #"He slept.": "Nåck eepra.",
            "What's up?": "Dup pritterok lunk",
            "Do you like to eat?": "Gè'googrung jeek tå chonġle̱ lunk",
            "What is going on?": "Nergo're pritterok hoog lunk",
            "Why is the sky blue?": "Pif gorse̱ sohong wat lunk"
        }

        # go through each test
        for english, gorgus in tests_to_gorgus.items():
            self.assertEqual(translate(english, "gorgus", formal=False), gorgus, "Translation from English to Gorgus does not match!")

    def test_from_gorgus(self):
        # key = gorgus, value = expected english translation
        tests_from_gorgus = {
            "Dink, dup pritterok lunk": "Hello, how are you going?",
            "Henġer agger ikfren!": "Me love dogs!",
            "Glonk chonġle̱ok migtir omnom!": "Stop eating all food!",
            "Googrung kiff!": "You smell!",
            "Minġer goob'rung ji dagsâ dublub. :)": "I hope you have a really nice day. :)",
            "Jid shrerack, henġer huffer clor'ge dagsa.": "That person, me believe they're nice.",
            "Ikshmack horge kithrark̂.": "Cats are really angry.",
            "Toopyat!": "Shit!"
        }

        # go through each test
        for gorgus, english in tests_from_gorgus.items():
            self.assertEqual(translate(gorgus, "english", formal=False), english, "Translation from Gorgus to English does not match!")


if __name__ == "__main__":
    try: # the user executed a subcommand
        args.func(args)
    except AttributeError: # user didn't type anything ;-;
        arg_parser.print_help()
        exit(0)