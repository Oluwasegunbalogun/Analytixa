# Defects and gaps found by the benchmark

Recorded in the order they were found. Interpreter vocabulary was frozen before the semantic
benchmark ran and was NOT tuned to its questions, so its errors stay in the results.

| # | Found by | Defect or gap | Action |
|---|---|---|---|
| 1 | Review of v3 (before the benchmark) | External-effect idempotency was kept only inside one run's saved state. A crash between the effect and the state save, or a second run on the same item, could repeat the effect. | Added the action_effects table. The effect and its record commit in one transaction. Tests RC1, RC2, IP1 to IP3. |
| 2 | Review of v3 | An approval was not re-checked when used. A deactivated approver, a changed role, an old approval or a changed policy did not invalidate it. | Added approval re-validation at the moment of use, with a 24-hour validity window in the autonomy policy. Tests AP1 to AP5. |
| 3 | Review of v3 | Ledger verification could not detect truncation of the newest entries or a full rewrite with recomputed hashes. | Added ledger anchors and anchored verification. Tests AU5 and AU6 report both results. |
| 4 | Credit boundary test CB26 | When the default probability was above the decline band, R9 appeared twice in the list of failed rules. The recommendation was correct. | Fixed in evaluate_rules. CB26 passes. |
| 5 | Semantic benchmark S010 | An answer over an empty result (approval rate with no decisions yet) crashed the narrator. | Fixed in bi_answer. |
| 6 | Semantic benchmark | Interpreter gaps (not fixed, reported as results): "loan count" is not a known phrase, "monthly income" adds a month breakdown, a second "and" dimension is dropped, "leaving out", "but" and "non-" are not read as exclusion, date ranges and "past week" are not read, and thresholds or contradictions are answered instead of refused. | Reported in Chapter Five. These cases are what the interpretation check and correction are for. |
