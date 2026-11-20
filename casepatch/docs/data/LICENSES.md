# Name-list licenses

Files in this directory are used only as generic given-name gazetteers for detecting
person names. They are not case-specific and are not tuned on holdout documents.

| File | What it is | License | Source |
|---|---|---|---|
| `given_names_us.txt` | US given names with at least 0.01% of US births in some year 1880–2008 | Public domain (US federal government work) | US Social Security Administration baby-name data, via [hadley/data-baby-names](https://github.com/hadley/data-baby-names) |
| `given_names_intl.txt` | Latin-script / romanized given names from two CC0 collections, minus names already in the US list. Wikidata entries that are common English words are omitted unless they also appear as a US SSA or popular-by-country given name. Rebuilt by `tools/build_given_names_intl.py`. | CC0 1.0 | [sigpwned/popular-names-by-country-dataset](https://github.com/sigpwned/popular-names-by-country-dataset) (popular given names, 106 countries; itself compiled from Wikipedia "most popular given names" lists); [Wikidata given names](https://www.wikidata.org/wiki/Wikidata:Licensing) extracted by [brawer/wikidata-names](https://github.com/brawer/wikidata-names) and published at [names.toolforge.org](https://names.toolforge.org/) (`givennames.csv.gz`) |

The English-word filter used only when *building* `given_names_intl.txt` is the MIT-licensed [google-10000-english](https://github.com/first20hours/google-10000-english) list. That word list is not shipped in the runtime package.
