import nltk
import time

unigram_tagger_file = "brown_unigram_tagger_model.pkl"
bigram_tagger_file = "brown_bigram_tagger_model.pkl"

def nltk_download(packagePath, package) -> bool:
    try:
        print(f"Checking for {package}...")
        nltk.data.find(packagePath)
        return True
    except LookupError:
        print(f"Downloading {package}...")
        #console.print(
            #f"[bold orange1]Warning![/bold orange1] {package} was not found! Attempting to automatically install..", highlight=False)
        #console.print(
            #"[dim]Disabling SSL check to prevent issues on certain operating systems (MacOS, I'm looking at you)...[/dim]", highlight=False)
        import ssl
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        download_success = nltk.download(package)
        return download_success

def generate_unigram_tagger():
    print("Training unigram tagger...")
    taggerStartTime = time.perf_counter_ns()

    unigram_tagger = nltk.UnigramTagger(nltk.corpus.brown.tagged_sents())

    taggerEndTime = time.perf_counter_ns()
    print(f"Training took {(taggerEndTime - taggerStartTime) / 1000000000}s.")

    return unigram_tagger

def generate_bigram_tagger():
    print("Training bigram tagger...")
    taggerStartTime = time.perf_counter_ns()

    bigram_tagger = nltk.BigramTagger(nltk.corpus.brown.tagged_sents())

    taggerEndTime = time.perf_counter_ns()
    print(f"Training took {(taggerEndTime - taggerStartTime) / 1000000000}s.")

    return bigram_tagger

def import_unigram_tagger():
    try:
        import dill
        import os

        # noinspection PyBroadException
        #try:
        with open(f'{os.path.dirname(__file__)}/{unigram_tagger_file}', 'rb') as fin:
            return dill.load(fin)
    # pickling doesn't work in the WASM version of CPython, prevent dill loading errors from borking the web translator.
    # Also the file may not exist.
    #     ~ HSI
    except:
        return None
    #except:
        #return None

def import_bigram_tagger():
    try:
        import dill
        import os

        # noinspection PyBroadException
        #try:
        with open(f'{os.path.dirname(__file__)}/{bigram_tagger_file}', 'rb') as fin:
            return dill.load(fin)
    # pickling doesn't work in the WASM version of CPython, prevent dill loading errors from borking the web translator.
    # Also the file may not exist.
    #     ~ HSI
    except:
        return None
    #except:
        #return None

def get_unigram_tagger_and_train_if_not_found():
    unigram_tagger = import_unigram_tagger()

    if unigram_tagger is None:
        print("Unigram tagger not found, training...")

        unigram_tagger = generate_unigram_tagger()
    else:
        print("Unigram tagger found!")

    return unigram_tagger

def get_bigram_tagger_and_train_if_not_found():
    bigram_tagger = import_bigram_tagger()

    if bigram_tagger is None:
        print("Bigram tagger not found, training...")

        bigram_tagger = generate_bigram_tagger()
    else:
        print("Bigram tagger found!")

    return bigram_tagger

def main():
    import dill
    import os

    brown_download_success = nltk_download("corpora/brown.zip", "brown")
    if not brown_download_success:
        print("Failed to download the Brown corpus - which is required for training. You will need to download it manually.")

    unigram_tagger = generate_unigram_tagger()
    bigram_tagger = generate_bigram_tagger()

    print("Saving taggers...")
    with open(f'{os.path.dirname(__file__)}/{unigram_tagger_file}', 'wb') as fout:
        dill.dump(unigram_tagger, fout)

    with open(f'{os.path.dirname(__file__)}/{bigram_tagger_file}', 'wb') as fout:
        dill.dump(bigram_tagger, fout)

    print("Complete!")

if __name__ == "__main__":
    main()