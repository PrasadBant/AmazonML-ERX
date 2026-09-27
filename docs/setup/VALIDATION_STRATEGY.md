# Validation Strategy — Amazon ML Challenge 2026

**Status: INFERRED recommendation**, grounded in CONFIRMED dataset facts from
`DATASET_INVENTORY.md`, `GROUND_TRUTH_ANALYSIS.md`, and `LEAKAGE_AUDIT.md`. No
validation split has been created yet — this document only proposes the design.

## Why a naive random row split is unsafe

A random split of `train_ground_truth.tsv` rows into train/val would randomly assign
Source 1 entities to each side, which is safe **for the S1 side** (each S1 entity's
label is independent of every other's — CONFIRMED 1:1 row correspondence, no
duplicate S1 rows). But it is not safe **for the S2/S3 side** unless handled
carefully:

- A single Source 2 or Source 3 record could plausibly be a genuine duplicate/near-
  duplicate of another record in the same file (CONFIRMED 0.43-0.73% exact-duplicate
  rate in S2/S3 — see `DATA_PROFILE.md`). If a near-duplicate pair happens to land as
  the true match for two different S1 entities that get split across train/val, a
  blocking or matching model could see one twin during training and trivially "solve"
  the other twin at validation time — inflating the validation score.
- Any candidate-generation (blocking) step built on the *full* S2/S3 pool must be
  evaluated using a candidate pool that mirrors deployment: at test time, the model
  will block against the *entire* test S2/S3 file, not a filtered subset. If validation
  blocks only against a held-out slice of S2/S3, recall-at-blocking will look better
  than it will in production.

## Recommended design

1. **Split at the Source 1 entity level, not the row level.** This is what a random
   row split on the ground truth already does correctly (each row = one S1 entity), so
   the entity-level unit of the split is fine. The risk is in what stays visible on the
   S2/S3 side, not in how S1 entities are chosen.

2. **Do not remove S2/S3 records from the pool when a matching S1 entity is held out.**
   Keep the full `train_source2.tsv` / `train_source3.tsv` as the candidate pool for
   both the training-fold and validation-fold S1 entities. Only the *labels* (which S1
   entity matches which) are split; the searchable universe of S2/S3 records is not
   partitioned. This mirrors production, where the model always blocks against the
   complete test S2/S3 files.
   - Consequence: a matching classifier trained on (S1, S2/S3) candidate pairs from the
     training-fold S1 entities could technically see S2/S3 records that also happen to
     be true matches for validation-fold S1 entities, as *negative* examples paired with
     other S1 entities. This is a mild, unavoidable form of information sharing given
     the task structure (S2/S3 pools are shared, not partitioned, in both the official
     train and test sets) — INFERRED to be acceptable because it mirrors exactly how
     the official test set works (test S1 entities are also matched against a shared,
     unpartitioned S2/S3 pool). Partitioning S2/S3 by fold would create a mismatch with
     how the model will actually be scored.

3. **Stratify the split by country.** Country distribution shifts materially between
   train and test (CONFIRMED, `DATA_PROFILE.md`: US 60%→38%, India 40%→47%, France
   0%→15%). A random split of training S1 entities will preserve train's own
   US/India mix (~60/40), which will *not* resemble the test distribution. Two options,
   in order of preference:
   - **Preferred:** stratify the held-out validation entities toward a higher India
     share (and expect it will still not include France, which is fundamentally
     untestable from training data). This gets validation scores closer to what the
     India-heavy, 3-country test distribution will show, without literally being able
     to simulate France.
   - **Minimum bar:** at least report validation F_0.5 broken out by country (US vs.
     India) rather than a single pooled number, so a country-driven regression doesn't
     hide inside an averaged score. France remains an explicit blind spot to flag in
     the methodology doc, not something to silently assume away.

4. **Match-count stratification.** The ground truth's match-count distribution is
   right-skewed (median 3, 5.6% singletons — CONFIRMED, `GROUND_TRUTH_ANALYSIS.md`).
   A validation split should preserve this distribution (e.g. stratified sampling on
   `len(matched_entity_ids)` bucketed as {0, 1, 2-3, 4-5, 6+}) so validation doesn't
   accidentally over- or under-represent singletons, since singleton handling is
   explicitly scored (F_0.5 = 1.0 for a correctly predicted empty list) and easy to get
   wrong in either direction.

5. **Run the cross-split leak checks from `LEAKAGE_AUDIT.md` §3 and §6 against the
   drawn train/val subsets**, not just against the official train/test files, before
   trusting any validation number.

## Recommended split ratio

INFERRED, not yet tuned: an 85/15 or 80/20 split of Source 1 entities (stratified by
country and match-count bucket as above) gives roughly 330K-440K validation entities
out of 2.2M — large enough for a stable F_0.5 macro-average, small enough to leave the
majority of labeled data for training/threshold-tuning.

## What this document does not yet cover

- The exact blocking/candidate-generation method (TF-IDF, phonetic keys, LSH, etc.) —
  out of scope for setup; to be decided during actual modeling.
- Whether multiple validation folds (k-fold) are worth the extra compute vs. a single
  held-out split — deferred until first-pass timing data exists (see
  `COMPUTE_STRATEGY.md`).

---

## Phase 2: as implemented (CONFIRMED)

- `workspace/src/er/folds.py` assigns each train S1 entity to fold = md5(entity_id) mod 10. Measured balance across all 10 folds: 40.0±0.2% India, 5.5–5.6% singletons, and 3.46 average matches per fold.
- **Model training:** blocked candidates of S1 sampled from folds 0–7 (80K, 160K or 120K entities depending on the experiment).
- **Validation:** 50K S1 from fold 9, blocked against the **full** train S2+S3 pool. The pool is shared, as recommended above.
- **Fold 8:** untouched spare.
- **Leakage control for learned artifacts:** the Indic→Latin transliteration dictionary (`translit.py`) is learned only from label pairs of folds 0–7.
- **Metric:** `metrics.py` computes F0.5 per S1 entity and averages over all validation S1, including singletons. A correct empty prediction scores 1.0, and any prediction on a singleton scores 0. This mirrors the README exactly. Micro precision and recall are reported only as diagnostics.
- **By-country reporting:** implemented for every model (India / US). France cannot be validated (test-only). The model excludes the `country` feature, which cost only 0.0006 F0.5, so France is scored by the same country-agnostic function.
- **One-owner post-processing:** validated with `validate_owner.py`. Competing S1 entities from *all* other folds that rank a contested record in their top-20 are scored with the full pipeline, so the constraint is evaluated with realistic competition rather than only within the validation sample.
