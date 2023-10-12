"""
"""
import os
from pathlib import Path
from urllib import parse
from nltk.tokenize import sent_tokenize

import nlp_utils as nlpu

localfolder = "texts"
verbose = 2

# if lines.cfg file exists, do nothing, else create one
if not os.path.exists(os.path.join(localfolder, "lines.cfg")):
    CFG = """fname,start,end
    alice28-1476.txt,104,3704
    wizoz10-1740.txt,310,5100"""
    # write content of CFG into f'{localfold}/lines.cfg' file
    with open(os.path.join(localfolder, "lines.cfg"), "w") as f:
        f.write(CFG)

# load cfg file and build a dictionary of the form: {fname: (start, end)}
with open(os.path.join(localfolder, "lines.cfg"), "r") as f:
    lines = f.readlines()
    lines = [line.strip().split(",") for line in lines[1:]]
    config = {line[0]: (int(line[1]), int(line[2])) for line in lines}

if verbose:
    print("text config:", config)

f1name = Path(parse.urlsplit(nlpu.ALICE_URL).path).name  # "alice28-1476.txt"
f2name = Path(parse.urlsplit(nlpu.WIZARD_URL).path).name  # "wizoz10-1740.txt"


def load_source_text(fname, localfolder="texts", skip_header=False):
    # load alice-text from localfold
    with open(os.path.join(localfolder, fname), "r") as f:
        if skip_header:
            # and get only target lines (readlines 104:3704)
            start, end = config[fname]
        else:
            start, end = None, None
        text = "".join(f.readlines()[slice(start, end, None)])
    return text


try:
    alice_text = load_source_text(fname=f1name, skip_header=True)
    wizoz_text = load_source_text(fname=f2name, skip_header=True)
except FileNotFoundError:
    nlpu.download_text(nlpu.ALICE_URL, localfolder)
    nlpu.download_text(nlpu.WIZARD_URL, localfolder)
    # load texts
    alice_text = load_source_text(fname=f1name, skip_header=True)
    wizoz_text = load_source_text(fname=f2name, skip_header=True)

if verbose == 2:
    print("files in ./texts/\n", os.listdir("./texts/"))

corpus_alice = sent_tokenize(alice_text)
corpus_wizoz = sent_tokenize(wizoz_text)

if verbose:
    print("Num. of sentences in texts:", len(corpus_alice), len(corpus_wizoz))
    if verbose == 2:
        print(corpus_alice[2])
        print(corpus_wizoz[30])


def sentence_tokenize(
    fname,
    quote_char="\\",
    sep_char=",",
    include_header=True,
    include_source=True,
    extensions=("txt"),
    **kwargs,
):
    """cleans line breaks in text file, and save to CSV files"""
    source = load_source_text(fname, skip_header=True)
    contents = source.replace("\n", " ").replace("\r", "")
    corpus = sent_tokenize(contents, **kwargs)

    # builds a CSV file containing tokenized sentences,
    # if include_header is True, and include_source also is true,
    # then using header: ('sentence,source\n'), else ('sentence\n')
    # then write the sentence in corpus in form of:
    # f"{quote_char}{sentence}{quote_char}{sep_char}{fname}"
    base = os.path.splitext(fname)[0]
    new_fname = f"{base}.sent.csv"
    new_fname = os.path.join(localfolder, new_fname)
    with open(new_fname, "w") as f:
        # Header of the file
        if include_header:
            if include_source:
                f.write("sentence,source\n")
            else:
                f.write("sentence\n")
        for sentence in corpus:
            if include_source:
                f.write(
                    f"{quote_char}{sentence}{quote_char}{sep_char}{fname}\n"
                )
            else:
                f.write(f"{quote_char}{sentence}{quote_char}\n")

    return new_fname


# to get file's path that contain 'sent' in filename in localfolder
def get_file_path(localfolder):
    return [
        os.path.join(localfolder, f)
        for f in os.listdir(localfolder)
        if "sent" in f
    ]


new_fnames = get_file_path(localfolder)
if len(new_fnames) == 0:  # if no such files, create them
    new_fnames = []
    for fname in (f1name, f2name):
        new_fname = sentence_tokenize(fname=fname)
        new_fnames.append(new_fname)
if verbose:
    print(new_fnames)
