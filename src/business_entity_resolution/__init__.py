"""Business entity resolution for the Amazon ML Challenge 2026.

Pipeline: load -> normalize -> block (candidate generation) -> pairwise features -> XGBoost matcher
-> threshold + one-owner decision -> matching_results.tsv / candidate_pairs.tsv.
"""
