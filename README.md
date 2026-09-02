# pykarsin

Finds near-duplicate and potentially irrelevant entries in a qualitative coding codebook exported to CSV.

The tool is codebook-agnostic. It does not depend on the software that produced the CSV and never modifies the source file.

## Caveats

Lexical similarity is not semantic similarity.

pykarsin compares character n-grams, not meaning. Two codes may score highly while referring to different concepts, such as anticipated and experienced harm. Conversely, semantically similar codes may score poorly if their wording differs substantially.

All flagged items require manual review against the original source excerpts. The output is a review queue, not a deletion list.

The tool runs fully offline and deterministically. It makes no network calls, and identical input and settings produce identical output.

## Install

```bash
pip install .
```

For isolated CLI use:

```bash
pipx install .
```

## Usage

```bash
pykarsin scan your_codebook.csv
```

When scanning a new file, pykarsin displays the detected columns and asks for confirmation before continuing.

```bash
pykarsin scan tests/fixtures/example_codebook.csv --dry-run
```

Use `-y` or `--yes` to accept the detected column mapping without prompting. This is useful for scripting.

Column mapping can also be set manually with `--code-col`, `--comment-col`, and `--group-col-prefix`.

Some exports store group membership across several separate columns without a common header prefix. Atlas.ti, for example, may use `Code Group 1` through `Code Group 50`. In these cases, use `--group-cols` with a range or list of column numbers:

```bash
pykarsin scan your_codebook.csv --group-cols 3-52
pykarsin scan your_codebook.csv --group-cols 3,5,9
```

Examples:

```bash
pykarsin scan tests/fixtures/example_codebook.csv --yes
pykarsin scan your_codebook.csv --yes --export csv
pykarsin scan your_codebook.csv --yes --export markdown
pykarsin scan your_codebook.csv --yes --export html
pykarsin scan your_codebook.csv --yes --relevance research_questions.txt
```

`--export html` writes a single self-contained HTML file without external fonts, scripts, or network calls. It includes short explanations for each section and is intended for manual review of merge and retention candidates.

All export formats include relevance results when `--relevance` is used.

Run:

```bash
pykarsin scan --help
```

for the full flag list.

## Input format

pykarsin expects a CSV with, by convention:

* a **Code** column, required
* a **Comment** column, optional
* zero or more **Code Group N** columns, optional

Rows whose comment contains `merged with` are treated as already superseded and excluded by default. Use `--include-merged` to include them.

Header matching is case-insensitive. The tool does not assume that the file was produced by any specific software.

Semicolon-delimited CSV files and UTF-8 BOM are handled automatically.

## How it works

Similarity is calculated using character n-gram TF-IDF with `analyzer="char_wb"` and n-grams of 3 to 5 characters, followed by cosine similarity.

This approach is relatively robust to Finnish inflection and compound words without requiring lemmatization or other language-specific NLP.

Two passes use the same similarity matrix:

* **Pairwise pass** (`--pair-threshold`, default `0.6`) identifies stronger candidate pairs.
* **Clustering pass** (`--cluster-threshold`, default `0.5`) uses union-find to form looser transitive groups. If A is sufficiently similar to B and B to C, all three are placed in the same cluster even if A and C are less similar.

Parent-child pairs are excluded from both passes. pykarsin recognises the colon-prefix convention:

```text
Parent
Parent: Child text
```

A child is expected to resemble its parent, so this similarity is not reported as a duplicate candidate.

If exactly one code in a pair contains a Finnish negation marker such as `ei` or `eikä`, the pair is placed in a separate **check polarity** section. Similar wording may otherwise obscure opposite meaning.

Clusters containing four or more codes are marked for possible chaining. A shared word stem can connect several loosely related codes through transitive similarity. The tool flags these clusters but does not classify them automatically.

## Relevance checking

The optional `--relevance` pass compares each code with research questions supplied in a text file.

By default, it flags the bottom 10% of codes by similarity using `--relevance-percentile`, together with codes that are too short to provide a reliable n-gram signal.

These codes are labelled **needs human read**, not irrelevant.

Relevance is ranked relative to the current codebook rather than evaluated against an absolute similarity threshold. This is important for deductive and theory-driven codebooks, where relevant codes may have little lexical overlap with the research questions.

A keyword-absence approach was tested but produced too many false positives.

## Limitations

* Similarity is purely lexical. The tool does not perform semantic disambiguation, Finnish-specific NLP, or lemmatization.
* Relevance scores are relative to the current codebook and should not be interpreted as absolute measures of relevance.
* Transitive clustering may produce chaining through shared word stems. These cases are flagged but not resolved automatically.
* The tool only reads and reports. It never modifies the source CSV or writes back to the original codebook software.

## License

EUPL-1.2. See [LICENSE](LICENSE).
