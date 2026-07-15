# pykarsin

Finds near-duplicate and possibly irrelevant entries in a qualitative-coding
codebook exported to CSV. Codebook-agnostic — it doesn't know or care which
tool the CSV came from, and it never writes back to the source file.

## Caveats, read this before trusting the output

- Lexical similarity is not semantic understanding. The tool compares
  character n-grams, not meaning. Two codes can score high while meaning
  different things (e.g. anticipated vs. experienced harm), and two codes
  can mean the same thing while scoring low (very different wording).
- Every flagged item needs manual review against the original source
  excerpts. The output is a review queue, not a deletion list.
- Runs fully offline and deterministically — no network calls, same input
  always gives the same output.

## Install

```
pip install .
```

or, for isolated CLI use:

```
pipx install .
```

## Usage

```
pykarsin scan your_codebook.csv
```

On first run against a new file, pykarsin shows you the columns it detected
and asks you to confirm the mapping before doing anything else:

```
pykarsin scan tests/fixtures/example_codebook.csv --dry-run
```

Pass `-y`/`--yes` to accept the auto-detected columns without asking
(useful for scripting), or `--code-col`/`--comment-col`/`--group-col-prefix`
to override the detected mapping directly. For exports that spread group
membership across many per-slot columns with no shared header prefix (e.g.
Atlas.ti's `Code Group 1`..`Code Group 50`), use `--group-cols` with a
column range instead, e.g. `--group-cols 3-52` or `--group-cols 3,5,9`.

```
pykarsin scan tests/fixtures/example_codebook.csv --yes
pykarsin scan your_codebook.csv --yes --export csv
pykarsin scan your_codebook.csv --yes --export markdown
pykarsin scan your_codebook.csv --yes --export html
pykarsin scan your_codebook.csv --yes --relevance research_questions.txt
```

`--export html` writes a single self-contained file (no external fonts,
scripts, or network calls) meant to be opened directly in a browser — it
includes a plain-language legend for each section, so it's usually the
easiest format to read through when deciding what to merge or keep. All
three export formats include the `--relevance` results too, when used.

Run `pykarsin scan --help` for the full flag list.

## Input format

A CSV with, by convention:

- a **Code** column (required),
- a **Comment** column (optional — rows whose comment contains
  `merged with` are treated as already-superseded and excluded by default,
  use `--include-merged` to include them),
- zero or more **Code Group N** columns (optional, sparse).

Header names are matched case-insensitively; pykarsin doesn't assume a
specific export tool wrote them. Semicolon-delimited CSV (common from
Finnish-locale Excel) and a UTF-8 BOM are both handled automatically.

## How it works

Similarity is character n-gram TF-IDF (`analyzer="char_wb"`, n-grams of
size 3–5) plus cosine similarity — robust to Finnish inflection and
compounding without a lemmatizer. Two passes run over the same similarity
matrix:

- **Pairwise pass** (`--pair-threshold`, default `0.6`): high-precision
  candidate pairs.
- **Clustering pass** (`--cluster-threshold`, default `0.5`, union-find):
  looser, transitive grouping — if A~B and B~C both clear the threshold,
  they land in one group even if A~C alone wouldn't.

Both passes skip parent/child pairs (colon-prefix convention: `"Parent"` /
`"Parent: Child text"`), since a child is expected to resemble its parent
without being a duplicate.

Pairs where exactly one side carries a Finnish negation marker (`ei`,
`eikä`, ...) are routed to a separate "check polarity" section instead of
being ranked as high-confidence — similar wording, opposite meaning.
Clusters of 4 or more codes carry a warning that they may be chained
together through a shared word stem rather than actually belonging
together; this is a label, not an automatic classification, since telling
real duplication apart from stem-chaining needs a human reading the text.

The optional `--relevance` pass compares codes against user-supplied
research questions and flags the bottom `--relevance-percentile`% of codes
by similarity (default 10%), plus any code too short for a reliable
n-gram signal, as "needs human read" — not as a keyword-absence filter,
which was tried and produced too many false positives on real codebooks.
It's a relative ranking within your own codebook, not an absolute
cutoff — see Limitations below for why.

## Limitations

- Purely lexical: no semantic disambiguation, no Finnish-specific NLP or
  lemmatization pipeline. This is why the relevance pass ranks codes
  relative to each other instead of against a fixed similarity cutoff —
  deductive/theory-driven codebooks often have almost no literal wording
  overlap with the research questions even when a code is clearly
  on-topic, so an absolute threshold ends up flagging nearly the whole
  codebook.
- Cluster chaining artifacts (a shared compound stem pulling unrelated
  codes together) are flagged, not resolved — needs manual read.
- Only reads and reports; it never modifies the source CSV or writes back
  to any codebook tool.

## License

EUPL-1.2, see [LICENSE](LICENSE).
