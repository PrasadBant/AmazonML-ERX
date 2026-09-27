# output/

The challenge requires the final package to contain `output/matching_results.tsv` and
`output/candidate_pairs.tsv`. This repository writes every run to its own folder instead
(`submissions/<version>/output/`), so earlier submissions are never overwritten. To build a package,
copy the chosen version's two files here or into the zip's `output/` folder.

| File | Columns | Rule |
|---|---|---|
| `matching_results.tsv` | `source1_entity_id` `\t` `matched_entity_ids` | one row per test Source-1 id; comma-separated S2/S3 ids; empty = no match |
| `candidate_pairs.tsv` | `source1_entity_id` `\t` `candidate_entity_ids` | the final candidate set scored by the model; every match must appear here |

Check both files with `python scripts/validate_submission.py <folder> --official`.
