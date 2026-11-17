# Analytixa v4

Analytixa is a working prototype of HAARA, an eight-layer Human-in-the-loop Agentic Analytics Reference Architecture. It runs a fictional Nigerian bank (Lagoon Trust Bank by default). Agents plan analytical work and use governed tools. A rule-based oversight layer decides which steps run alone and which must wait for a named person. The bank sets its own loan, AML, retention, cash, fairness and autonomy policies inside the application.

All customers, accounts, transactions and applications are synthetic. No real bank data or policy is used.

## What is new in v4

- Your data. Anyone whose role the bank allows can upload a spreadsheet (.csv or .xlsx) and ask questions about it in plain words. The same oversight applies as for the bank's own data. See "Analysing your own data" below.
- Full analysis reports and interactive dashboards on uploaded data. See "Full analysis" and "Dashboards".
- The Evaluation tab under How the system works shows the results of the reproducible benchmark.

## What was new in v3

- Data lives in PostgreSQL. A free Supabase project works, and so does any Postgres 14 or later.
- The bank is a relational model with 22 tables, keys, foreign keys, money as numeric and dates as date. The trained models, policy versions, runs, decisions and the hash-chained ledger are stored there too.
- Staff sign in with their own accounts. Each person has one role. Every decision records their name, for example "Hauwa Bello (Credit manager)".
- A setup step asks for the database, the bank name and the staff you want.
- A simpler interface. Each person sees a Home page with what is waiting for them, and each task page shows one thing at a time.

## 1. Create the database (Supabase)

1. Sign in at supabase.com and create a new project. Keep the database password you choose.
2. Open the project and click Connect at the top.
3. Under Connection string, choose Session pooler and copy the URI. It looks like this:
   `postgresql://postgres.abcdefgh:[YOUR-PASSWORD]@aws-0-eu-west-2.pooler.supabase.com:5432/postgres`
4. Replace `[YOUR-PASSWORD]` with your database password.

Use the Session pooler string. The direct connection string needs IPv6, which many home and office networks do not have.

## 2. Install and set up

1. Install Python 3.10 or later.
2. Windows: double-click `start.bat`. macOS or Linux: run `./start.sh`.

The first start installs the packages and then runs `setup.py`, which asks four things:

| Step | What it asks |
|---|---|
| Database | The connection string from step 1. It tests the connection before going on. |
| Company | The bank name and the schema name in the database. |
| Building the bank | Nothing. It creates the tables, loads the synthetic bank and trains the models. This takes one to three minutes on Supabase. |
| Staff accounts | The administrator's name, email and password. Then it offers a demonstration team with one person per role, and lets you add more people one by one. |

Setup saves your answers in `config.env`. That file holds your database password, so do not share it.

When setup finishes, the server starts. Wait for the line that starts with "Ready", then open http://127.0.0.1:4000 and sign in.

Later starts skip setup. To run it again, use `python setup.py`.

### Demonstration team

If you accept the demonstration team, setup creates these accounts with the password you choose:

| Role | Name | Email |
|---|---|---|
| Analyst | Tola Adeyemi | tola@<bank>.test |
| Loan officer | Chidi Okafor | chidi@<bank>.test |
| Credit manager | Hauwa Bello | hauwa@<bank>.test |
| Compliance officer | Kunle Lawal | kunle@<bank>.test |
| Retail manager | Ngozi Eze | ngozi@<bank>.test |
| Branch operations | Musa Garba | musa@<bank>.test |

`<bank>` is the bank name in lower case without spaces, for example `lagoontrustbank`. Setup prints the exact emails.

The administrator can add people, change roles, turn accounts off and set passwords on the Staff page.

## Roles

| Role | Decides |
|---|---|
| Analyst | Plans and report releases for questions they ask |
| Loan officer | Loans within delegated authority, with a discussion with the credit agent |
| Credit manager | Credit exceptions and loans above the officer limit |
| Compliance officer | AML plans, payment holds above the limit, suspicious transaction reports |
| Retail manager | Retention offers and customer outreach |
| Branch operations | ATM cash orders above the limit |
| Administrator | Staff accounts, policies and language model settings |

## Who approves what

Some decisions belong to a role, such as a credit exception for any credit manager. Others belong to the person who asked. When the routing policy says "requester" (plan approval, data review, report release, escalation and the interpretation check), only the person who started the task sees and decides it. Another analyst cannot approve it for them. This holds for the administrator too: a decision on a task the administrator started goes to the administrator.

A decision waits until it is made. It shows in three places: on the task page itself, in Approvals, and in a strip under the top bar on every page ("2 decisions are waiting for you"). Each link in the strip opens the task at the decision. Moving to another page does not lose it. Deleting a dataset withdraws the open decisions of questions about it, and those tasks show as Cancelled.

## Checking how a question was understood

Every answer to a question about the bank starts with "How I understood your question", in plain words. The person who asked can confirm it or click "No, correct it". A correction can be typed in plain words ("I meant every city except Lagos") or set exactly with the measure, breakdown, period and filters, including "is not". The agent reruns the question with the correction, and the ledger records both the old and the new interpretation.

When a question has a condition such as not, except, outside or only, the agent asks the person to confirm the interpretation before it runs the query. The administrator sets this in Policies, Decision points, with the setting bi_interpretation_check: always, when_unsure (the default) or after_only.

## Analysing your own data

Open Your data, choose a file and click Upload and check. Analytixa then:

1. Reads the file and decides what each column is: a number to measure, a group, a date, an identifier or personal data.
2. Replaces personal columns (names, phone numbers, emails, BVN, account numbers, addresses) with tokens before anything is stored. Raw personal values never reach the database, and personal columns cannot be used in a question.
3. Checks the data and lists what to know before relying on an answer: duplicate rows, missing values, values that could not be read as numbers or dates, negative amounts, extreme values and dates in the future.
4. Records the upload in the audit trail with a fingerprint of the file.

Then ask a question, for example "Total outstanding by product for branches outside Lagos". The question runs as a task with five steps: check your data, interpret the question, query your dataset, verify the result, answer with the query used. As with questions about the bank, the answer opens with "How I understood your question", and the person who asked can confirm it or correct it in plain words ("No, I meant by branch, and leave out Kano too") or exactly in a form. When the wording has a condition such as not, except, above or top, the agent asks for confirmation before running the query.

Only the person who uploaded a dataset can see it, ask about it or delete it. The guard checks this on every tool call, not only in the interface. The bank sets who may upload and the size limits in Policies, Uploads.

A sample file is in `samples/sample_microfinance_loans.csv`: 1,803 loans of a fictional microfinance bank, with a few deliberate data problems for the check to find.

The questions the interpreter reads without a language model:

| Kind | Example |
|---|---|
| Totals, averages, medians, smallest, largest, counts | Average amount by product |
| Distinct counts | How many distinct branches are there? |
| Breakdowns, up to three | Outstanding by state and product |
| Time breakdowns | Amount by month, Count of loans by quarter |
| Filters and exclusions | Total amount in Lagos and Abuja, Amount by status excluding Closed |
| Number conditions | Number of loans with amount above 1 million by state |
| Periods | in 2025, in March, last 3 months, between March and June 2026 |
| Top and bottom | Top 5 branch by total amount |

| Several measures at once | Total sales, total profit and average profit margin by category |
| SQL-style measures | SUM(sales) and AVG(profit_margin) by region |

With a language model connected, the model reads the question first, and its reading is checked against the columns before use.

### Plans, versions and corrections

Each question becomes one plan: the measures with their aggregations, the breakdowns, the filters, the period, the sort and limit, and whether you asked for a table or an explanation. The SQL is compiled only from the plan and checked against it before it runs. A query that does not match its plan is not run.

Every correction makes a new plan version, listed under "Check how this answer was calculated". "Discard the previous plan" or "start again" replaces the whole plan. A correction that names the measures and the grouping replaces those and the filters. A shorter correction such as "leave out Furniture too" or "use region instead of category" changes only what it names. A correction in your own words is shown back to you for confirmation before it runs. Settings made exactly in the form run directly.

A confirmation applies to the plan version on your screen. If the plan changed in the meantime, the confirmation is refused and you review the new version. If your last correction and the plan disagree, for example you said "do not group by product name" and the plan still does, the plan cannot be confirmed until you correct it.

Tests: `python -m unittest tests.test_intent_state -v` (22 tests) and `python tests/test_uploads.py` (33 checks). `docs/intent-fix-report.md` explains the fix.

### Full analysis

A single question gives one query and one table. A request for a full analysis gives a report. Analytixa treats a request as a full analysis when it asks for several of these: overall totals, efficiency ratios, comparison between groups, rankings, trends, relationships, data quality, concentration, recommendations and a summary.

1. The agent builds a plan: the measures, the ratios you defined, the groups and items to compare, the time grains and the sections. A ratio written as "CTR = total clicks divided by total impressions, multiplied by 100" is read as SUM(clicks) / SUM(impressions) x 100.
2. You approve the plan before anything runs. The plan lists every formula. Metrics you asked for that the data cannot support, such as revenue or conversions, are named as unavailable.
3. The report is computed from every row, with no filters. Each figure shows its formula. Ratios are a ratio of totals. A ratio with a zero or missing denominator is left blank, never shown as 0. Overall ratios and averages of daily ratios are shown side by side.
4. Every total is checked against a SQL SUM on the stored table. A mismatch holds the report for your review.
5. Download report saves the whole report as a file.

Identifier columns (ad, account or customer numbers) are stored as text, so long numbers keep every digit.

Write a ratio as "NAME = total X divided by total Y, multiplied by 100. Lower is better." The last sentence sets which direction counts as good. Without it, a ratio whose top line is a cost or amount counts as lower is better, and any other ratio as higher is better. Only the number columns you name are totalled. If you name none, all of them are.

Tests: `python -m unittest tests.test_report -v` (13 tests). They use `tests/fixtures/ads_synthetic.csv`, which is invented data.

### Dashboards

Ask for a dashboard in plain words, for example: "Create a dashboard of Amount and Outstanding by Branch, State and Product, monthly by Disbursed On, with the top Loan ID. OSR = total Outstanding divided by total Amount, multiplied by 100. Lower is better."

The agent reads which columns to total, which ratios you defined, which columns to filter by, the time grain and the identifier to rank. It uses only the columns you name. If you name none, it picks sensible ones: every number column, the group columns with up to 40 values, the date, and the identifier with many values. With no grain named, it picks day, week or month from the date range.

The dashboard runs at once. It shows:

1. A row of filters: date from and to, one list per group column, and Show, which picks the figure every chart uses.
2. Tiles with the totals, the ratios and the row count. With a filter on, each tile shows its share of all rows.
3. A line over time with a hover readout, and a table view of the same figures.
4. A bar chart per group column. Click a bar to filter the whole dashboard by it. Click it again to clear it.
5. The top 10 by the chosen figure. Ratios are ranked only where the denominator is large enough, so tiny values do not top the list.
6. The data warnings from the upload check.

Download data saves the filtered figures as a CSV file.

The server sums the rows into a compact table by period and filter column. The page recomputes every figure from that table, so ratios stay ratios of sums under any filter. Totals are checked against SQL, and the table must add up to the totals and hold every row.

To change a dashboard, click "No, correct it" and say what to change: "add Status", "remove Gender", "only Amount", "show it weekly", or a new formula. Each change makes a new plan version, and you confirm it before it runs.

Tests: `python -m unittest tests.test_dashboard -v` (11 tests).

### The bank's own data

Your data also lists Bank data: five live, read-only datasets built from the bank's core tables. Ask questions, request a full analysis or create a dashboard on them exactly as on an uploaded file.

| Dataset | What it holds | Roles |
|---|---|---|
| Loan portfolio | Every loan with branch, region, product, segment, principal, outstanding, days past due, and 0 or 1 columns for default and arrears | Analyst, loan officer, credit manager |
| Loan applications | Applications with amount, collateral, declared income, bureau score and status | Analyst, loan officer, credit manager |
| Transactions | Cleaned transactions with channel, direction, category, branch, segment, inflow, outflow and a count | Analyst, compliance officer, retail manager |
| ATM cash | Daily withdrawals, replenishment and closing balance per ATM | Analyst, branch operations |
| Customer interactions | Contacts by channel, type and topic, with sentiment and resolution | Analyst, retail manager |

The administrator can use all five, and can change the roles in Policies, Uploads, with the setting bank_data_roles. Each dataset is a database view, so every question reads the current records. Names, phone numbers, income, BVN status and counterparties are left out. The count columns (Loans, In default, Transactions, Contacts and so on) let you define rates as formulas, for example "Default rate (NPL) = total In default divided by total Loans, multiplied by 100. Lower is better." Bank datasets cannot be deleted. The views are rebuilt each time Analytixa starts.

A full analysis or a dashboard typed on Ask goes to the matching bank dataset. Analytixa picks it from the topic words and the columns you name, among the datasets your role may use, so "Analyse the bank's loan portfolio ..." runs on Loan portfolio. If no dataset fits, Ask says which ones your role can use. Ordinary questions on Ask still go to the governed semantic layer.

Tests: `python -m unittest tests.test_bank_data -v` (7 tests) and `python -m unittest tests.test_serverless -v` (6 tests).

## The five use cases

| Use case | Start it from | Who decides |
|---|---|---|
| Loan decision | Loan applications page, Assess | Loan officer. Credit manager for exceptions. |
| Question about the bank | Ask page | Nobody for read-only answers. The analyst if the check is weak. |
| Customers likely to leave | Home or Ask | Analyst (plan, release), retail manager (offers, outreach) |
| Fraud and AML check | Home (compliance officer) | Compliance officer |
| ATM cash plan | Home (branch operations) | Branch operations for orders above the limit |

## Five-minute demonstration

1. Sign in as the loan officer. Open Loan applications and click Assess on APP-1010.
2. Read the recommendation, the failed rules and what would change the outcome.
3. In "Discuss with the credit agent", type: He has repaid his existing loan and his salary is now 1.2 million. The agent checks every rule again and answers.
4. Under Your decision, choose Approve, give a reason and click Sign the decision. The task now waits for a credit exception.
5. Sign out. Sign in as the credit manager. The item is on Home under Waiting for you. Approve the exception. The decision and memo are written.
6. Open the menu under your name and choose Audit trail. The chain check confirms that nothing was changed. Sign in as the administrator, open Policies and change R4 Debt-to-income. New tasks use the new version.

## The language model

The language model is optional. Without a key, every use case runs on rules and templates. To use one, sign in as the administrator, open the menu under your name, choose Language model settings and paste an OpenRouter key. A key pasted there stays in server memory only, so you paste it again after a restart. To keep it across restarts, add a line `OPENROUTER_API_KEY=your-key` to `config.env` (and `ANTHROPIC_API_KEY=your-key` for Anthropic). config.env is never packaged or committed. Analytixa never writes keys to the database or the audit trail.

When a model is connected, it reads each question first. Its reading is checked against the data before use, and the rules take over if the check fails.

Models are tried in order. The first model is tried, then each backup model in the list, then Anthropic if you add an Anthropic key. The next model is tried when one is busy, fails, times out after 25 seconds, returns nothing or returns output that cannot be read. Every switch goes into the audit trail as "model fallback", and the settings screen shows which model gave the last answer. The default backup list is poolside/laguna-s-2.1:free, nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free, openrouter/free, openai/gpt-4o-mini and deepseek/deepseek-chat. You can change it on the settings screen.

Tests: `python -m unittest tests.test_llm_fallback -v` checks the order, the switching, the Anthropic fallback and the rule fallback, using fake providers.

## How the layers map to code

| Layer | Where |
|---|---|
| L1 Data and provenance | `app/l1_data.py` (Postgres access, hash-chained ledger), `app/seed.py` (relational model and synthetic bank) |
| L2 Semantic and policy | `app/l2_policy.py` (versioned policy as code, semantic layer) |
| L3 Models | `app/l3_models.py` (credit, churn, anomaly and cash models, stored in the model registry table), `app/llm.py` |
| L4 Explainable AI | `app/l4_xai.py` |
| L5 Services and tools | `app/l5_tools.py`, `app/datasets.py` (uploaded datasets: storage, profile, interpreter and governed query) |
| L6 Agents and orchestrator | `app/l6_agents.py` |
| L7 Oversight control plane | `app/l7_control.py` |
| L8 Human interaction | `static/index.html`, `server.py`, `app/auth.py` (accounts, roles, sessions) |

## Database tables

Bank: `company`, `branches`, `products`, `customers`, `accounts`, `transactions` (with the `transactions_clean` view that removes replayed duplicates), `loans`, `loan_history`, `loan_applications`, `interactions`, `app_logins`, `atms`, `atm_cash`.

Simulated outside systems: `ext_loan_decisions`, `ext_cases`, `ext_holds`, `ext_str_filings`, `ext_campaigns`, `ext_cash_orders`.

Platform: `roles`, `app_users`, `sessions`, `policies`, `model_registry`, `runs`, `checkpoints`, `notifications`, `oversight_memory`, `dialogue`, `ledger`, `ledger_anchors`, `action_effects`, `datasets`.

Uploaded data: one table per dataset, named `uds_` followed by the dataset number.

Evaluation only: `_truth_churn`, `_truth_crime`, `_truth_seeded_errors`. Agents never read these.

Passwords are stored as salted PBKDF2 hashes. Session tokens are stored only as SHA-256 hashes.

## Evaluation

```
python evaluate.py
```

The benchmark rebuilds a fresh synthetic bank for each configuration and seed in its own schema, so live data is never touched. It has five parts: a comparative experiment of 48 scenarios under four configurations (autonomous, guards only with every policy check but no person, approve every step, HAARA) on seeds 1 to 5, 36 oversight robustness tests, 30 credit policy boundary cases, a 114-question semantic benchmark and an analytical validity report. Every expected outcome is written in `evaluation/benchmark/*.json` before the run. Results go to `evaluation/results/` and appear under How the system works, Evaluation. Use `--seeds 1` and `--parts oversight boundaries` to run some parts only, and `--modes guarded` to run one configuration. A full run takes about 12 minutes on a local database.

## When something goes wrong

Every button shows a spinner while it works, and a bar runs along the top of the page. If the server takes more than six seconds, a note says so. A free Supabase database that has been idle can take up to a minute to wake. While Analytixa runs, it sends a tiny query every 30 seconds to keep the connection warm, so clicks after a quiet spell stay fast.

If something fails, the message gives a reference such as E-1A2B3C. To see what happened:

1. Sign in as the administrator.
2. Open the menu under your name and choose Diagnostic log.
3. Click Copy, or Save as file, and paste the text into your report.

The log lists how long each request took, database reconnects, model switches, failed tasks and every error with its reference and full details. Passwords, API keys, session tokens and the database password are removed before anything is written. The same log is on disk at `logs/analytixa.log` (the five latest files of 1 MB are kept). Warnings and errors also appear in the server window.

## Public showcase on Vercel

Analytixa can run on Vercel's free plan as a public showcase. Visitors pick a role and look around without a password, and settings stay read-only. `docs/DEPLOY-VERCEL.md` explains each step. On Vercel (or with HAARA_SERVERLESS=1) there are no background threads. Each status check moves a running task forward, under a lease in the runs table. Start-up happens on the first request, and uploads are limited to 4 MB.

## Public showcase on Azure

With Azure for Students, Analytixa runs on Azure App Service with an Azure PostgreSQL database. `docs/DEPLOY-AZURE.md` explains each step. `python make_azure_zip.py` builds the upload without config.env, logs or tests. On App Service the app prepares itself as soon as the server starts, logs go to /home/LogFiles/analytixa, session cookies are marked secure, and the daily maintenance runs inside the app once a day.

## Maintenance

| Command | What it does |
|---|---|
| `python manage.py users` | Lists staff accounts |
| `python manage.py adduser` | Adds staff accounts |
| `python manage.py password EMAIL` | Sets a new password |
| `python manage.py verify` | Checks the ledger hash chain |
| `python manage.py reset` | Rebuilds the bank, policies, models and ledger. Staff accounts are kept. |

## Limits

- The data is synthetic, so detection and model results are cleaner than real data would give.
- Outside systems (loan origination, payment switch, financial intelligence unit gateway, CRM, cash-in-transit) are simulated tables.
- Real-time functions run on a replayed transaction window, not a live stream.
- Human decisions in the evaluation come from a scripted reviewer, not bank staff.
- The server is meant for one machine (127.0.0.1). Putting it on a network needs HTTPS in front of it.

## License

MIT. See `LICENSE`. All data the system generates is synthetic.
