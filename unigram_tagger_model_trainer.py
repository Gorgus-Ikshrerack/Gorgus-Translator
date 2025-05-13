import nltk
import time

tagger_file = "brown_ngram_tagger_model.pkl"

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

def generate_tagger():
    print("Training tagger...")
    taggerStartTime = time.perf_counter_ns()

    unigram_tagger = nltk.UnigramTagger(nltk.corpus.brown.tagged_sents())

    taggerEndTime = time.perf_counter_ns()
    print(f"Training took {(taggerEndTime - taggerStartTime) / 1000000000}s.")

    return unigram_tagger

def import_tagger():
    try:
        import dill
        import os

        # noinspection PyBroadException
        #try:
        with open(f'{os.path.dirname(__file__)}/{tagger_file}', 'rb') as fin:
            return dill.load(fin)
    # pickling doesn't work in the WASM version of CPython, prevent dill loading errors from borking the web translator.
    #     ~ HSI
    except ModuleNotFoundError:
        return None
    #except:
        #return None

def get_tagger_and_train_if_not_found():
    tagger = import_tagger()

    if tagger is None:
        print("Tagger not found, training...")

        tagger = generate_tagger()
    else:
        print("Tagger found!")

    return tagger

def main():
    import dill
    import os

    brown_download_success = nltk_download("corpora/brown.zip", "brown")
    if not brown_download_success:
        print("Failed to download the Brown corpus - which is required for training. You will need to download it manually.")

    unigram_tagger = generate_tagger()

    print("Saving tagger...")
    with open(f'{os.path.dirname(__file__)}/{tagger_file}', 'wb') as fout:
        dill.dump(unigram_tagger, fout)

    print("Complete!")

if __name__ == "__main__":
    main()