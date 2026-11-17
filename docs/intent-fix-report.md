# Intent resolution and correction adherence: investigation and fix

Scope: questions about uploaded datasets ("Your data", use case `upload`). Questions about the bank use the governed
semantic layer (`bi` use case) and a different interpreter. That path was checked for regressions and not changed.

## 1. How a question flows

| Step | Where | What it does |
|---|---|---|
| Ask | `server.py` `POST /api/runs` | Creates a run with `params.dataset_id` and `params.question` |
| n1 Check your data | `app/l6_agents.py` `ds_check` | Profile through tool D18 |
| n2 Interpret | `ds_intent` | Plan from tool M18 (language model, optional) or C12 (rules), interpretation gate |
| n3 Query | `ds_query`, tool D17 `app/datasets.py` `dataset_query` | Compiles SQL from the plan and runs it |
| n4 Verify | `ds_verify` | Checks on the result |
| n5 Answer | `ds_answer` | Writes the answer and the result record |
| Correct | `correct_interpretation` then `_correct_dataset` | Builds a new plan and reruns from n2 through `steer` |
| Confirm | `confirm_interpretation`, `decide` | Post-answer confirmation and the pre-run gate |

## 2. Root cause, with evidence

The bug reproduces on the 4.0 code with `tests/repro_intent_bug.py` and an orders file with the reported columns
(`tests/fixtures/orders.csv`). Function-level evidence from the 4.0 code:

1. **The plan could hold only one measure.** `interpret` returned `{"measure": {...}}`, a single aggregation. A request
   for total sales, total profit and average profit margin cannot be represented, so two of the three were dropped
   before any SQL existed. The language model path was validated against the same single-measure shape.
2. **The aggregation word was chosen by a fixed priority, not by position.** `_agg_word` tested patterns in the order
   count, average, median, total. In the reported question "total" appears at character 46 and "average" at 78, yet it
   returned `avg`. The measure column was the first number column named (`sales`), so the result was "Average sales".
3. **The first matching column in column order won.** For the phrase "product category", `_find` matched both
   `category` and `product_name` (via the phrase "product"). The loop took whichever came first in the file. With
   `product_name` before `category`, the plan grouped by `product_name`.
4. **Negated grouping added the column it negated.** "Do not group by product_name" matched the breakdown pattern
   "by product_name" and added `product_name`.
5. **Words about the output became filters.** "Show the results in a table" matched the product value "Table", so the
   plan gained the filter `product_name is Table`. Listing every category ("Office Supplies, Technology and Furniture")
   became a redundant filter on all three values.
6. **Corrections were merged into the old plan, never replacing it.** `_correct_dataset` started from a copy of the old
   intent and applied `filters.update(...)`. Nothing removed old fields. "Discard the previous query plan" had no meaning,
   so the stale `product_name is Table` filter survived every correction.
7. **A misread correction ran without confirmation.** Corrected plans were tagged `source="corrected"`, and `ds_intent`
   skipped the confirmation gate for that tag. A plain-words correction the parser misread therefore executed at once,
   while the acknowledgement "Understood. I will now answer ..." echoed the wrong plan.
8. **Confirmations were not bound to a plan.** The interpretation gate key was `interpretation:n2`, the same for every
   plan. `confirm_interpretation` recorded `ctx.params["intent"]` at the time of the click, with no plan version from
   the page, so a click on a page showing an older plan confirmed whatever the current plan was.

Hypotheses checked and **not** confirmed:

- SQL generation from stale state: `ds_query` compiled SQL from `ctx.params["intent"]` each time. The SQL always matched
  the (wrong) current plan.
- State restored by persistence, retries or handoffs: `steer` rewrote `params.intent`, cleared the result and reran from
  n2. No stale plan was restored. The plan was wrong when it was built, not when it was stored.

### The failed state transition

Question, then plan v1 = avg(sales) by category, filter product_name=Table (defects 1, 2, 5). Correction C1 merged into v1:
measure still avg(sales) (defects 1, 2), product_name added (defect 4), filter kept (defect 6). It ran without
confirmation (defect 7). Correction C2 ("discard the previous plan") merged again: dimension reset to category, the stale
filter kept, and a redundant three-value category filter added (defects 5, 6). The person was then asked to confirm a
plan nobody asked for, and the confirmation was not tied to the plan shown (defect 8).

## 3. The fix

One structured, versioned plan is the only source of truth after interpretation. All later steps read it, the SQL is
compiled from it and checked against it, and approvals name it.

| Defect | Change | File and function |
|---|---|---|
| 1 | Plan with `metrics` (a list), `dimensions`, `filters`, `period`, `sort`, `limit`, `grain`, `output_format`, `explain`, `version`, `notes`. The 4.0 shape is still accepted | `datasets.py` `validate`, `_normalise` |
| 2 | Each number column named gets the aggregation word nearest before it in the same clause. `SUM(x)` syntax is read. Ratios such as margins default to average | `_metrics` |
| 3 | The longest match wins, then the one that ends last ("product category" is `category`) | `_best`, `_mentions` |
| 4 | "Do not group by X", "not by X" and "remove X from the grouping" exclude X | `NOT_GROUP` in `interpret` |
| 5 | Phrases about the output ("in a table") are masked before values are matched. Listing every value of a column is not a filter, and a note says so | `OUTPUT_PHRASES` in `interpret` |
| 6 | Three correction modes: **replace** (explicit "discard", "start again", "replace the plan"), **restate** (the correction names measures and grouping, so those and the filters are replaced) and **update** (only what is named changes). "X instead of Y" swaps a column. Every correction creates version n+1 | `l6_agents.py` `_correct_dataset` |
| 7 | A plain-words correction is confirmed before it runs (unless the bank sets `after_only`). Exact form settings run directly | `ds_intent` |
| 8 | The gate key includes the plan version and hash. Approving a gate whose plan has changed is refused. Post-answer confirmation must send the plan version shown, and is refused for any other version | `ds_intent`, `decide`, `confirm_interpretation`, `_upload_plan_problem` |

Further safeguards:

- **SQL check before execution.** `check_sql` parses the compiled SQL independently and compares table, aggregations
  and aliases, grouping columns, filtered columns, period and limit with the plan. D17 refuses to run on any mismatch.
- **Grain check.** D17 counts the distinct groups for the same filters. `ds_verify` checks one row per group.
- **No stale downstream state.** `steer` clears the query result, checks and interpretation when the plan changes.
  `ds_verify` and `ds_answer` refuse a result whose `plan_hash` differs from the current plan.
- **Contradictions ask for clarification.** `constraints` extracts explicit instructions from the latest correction
  (named aggregations, excluded or "only" groupings, "no filters"). `conflicts` compares them with the plan. A gate
  showing conflicts cannot be approved, and words and form settings that disagree are refused with an explanation.
- **Audit.** New ledger events: `plan_created`, `interpretation_corrected` (with from and to versions, hashes and the
  mode), `correction_unchanged`, `correction_conflict`, `plan_confirmed`, `confirmation_refused` and `plan_executed`
  (with the version, SQL and parameters). Approvals record the plan version and hash they approved. The task page lists
  the plan versions.

Governance is unchanged: the same guard, scopes, prechecks, routing to the asker and approval re-validation apply. No
path marks a plan approved without the person who asked.

## 4. Before and after

Reproduction: `python tests/repro_intent_bug.py .` (question, correction C1, replace instruction C2).

| Step | 4.0 (before) | After the fix |
|---|---|---|
| Question | Average sales, by category, product_name is Table | Total sales, total profit and average profit_margin, by category (version 1) |
| SQL | `AVG("sales") ... WHERE "product_name" IN ('Table') GROUP BY "category"` | `SUM("sales") AS total_sales, SUM("profit") AS total_profit, AVG("profit_margin") AS average_profit_margin ... GROUP BY "category"` |
| Correction C1 | Average sales, by category and product_name, product_name is Table (ran without confirmation) | Same plan as version 1, so the agent says the correction changes nothing |
| Replace C2 | Average sales, by category, product_name is Table, category is Office Supplies or Technology or Furniture | Version 2, same measures by category, no filter, note that every category was listed. Asked to confirm, then ran |

With the reported wrong state injected as version 1, correction C1 produces exactly the expected plan as version 2
(test 05).

## 5. Tests

`python -m unittest tests.test_intent_state -v`: 22 tests, 12 unit (no database) and 10 integration (PostgreSQL).

| Required case | Tests |
|---|---|
| 1 total sales is SUM | `test_01` |
| 2 total profit is SUM | `test_02` |
| 3 average profit margin is AVG | `test_03`, `test_03b` |
| 4 category only, no product_name | `test_04`, `test_04b` |
| 5 explicit correction replaces | `test_05` |
| 6 repeated corrections | `test_06` |
| 7 confirmation applies to the plan shown | `test_07` |
| 8 changed plan invalidates SQL and approvals | `test_08`, `test_08b`, `test_sql_check_catches_mismatch` |
| 9 SQL from the latest approved plan | `test_09` |
| 10 no unintended filters | `test_10` |
| 11 contradiction asks for clarification | `test_11_conflict_detection`, `test_11_contradiction_asks_for_clarification` |
| 12 existing workflows | `test_12a`, `test_12b`, `tests/test_uploads.py` (33 checks), benchmark regression on seed 1 |

## 6. Limits and risks

- The rule-based interpreter covers the phrasings in the tests and the uploads test. Unusual phrasings can still be
  misread. The plan is always shown, a plain-words correction is confirmed before it runs, and the form sets any plan
  exactly.
- The language model path was not tested here, because no model key is configured in this environment. Its output is
  validated against the same plan structure and the same conflict check, but its reading quality is unmeasured.
- `check_sql` parses SQL produced by Analytixa's own compiler. It is a guard against compiler and state defects, not a
  general SQL validator.
- A correction such as "use the median instead" on a plan with several measures is refused with a request to name the
  measure, by design.
- Category value lists hold up to 200 values per column. For columns with more values, the "every value listed" rule
  does not apply.
