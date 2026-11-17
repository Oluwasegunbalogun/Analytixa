# Deploying Analytixa to Vercel as a public showcase

This puts Analytixa on a free public link. Visitors pick a role and look around without a password. All data is synthetic.

## What changes on Vercel

| On your computer | On Vercel |
|---|---|
| Tasks run in the background | Tasks move forward each time the page checks on them (about every second). A lease in the database stops two requests working on the same task. |
| Starts once and stays up | Starts on the first visit after a quiet spell, in a few seconds |
| Uploads up to 25 MB | Uploads up to 4 MB (Vercel's request limit is 4.5 MB) |
| Diagnostic log in logs/ | Diagnostic log in Vercel, under Logs |
| Sign in with a password | Showcase mode: "Try the demo" buttons, one per role |

In showcase mode, settings, policies and staff accounts are read-only. A daily job at 05:00 UTC:
- keeps the free Supabase database awake
- refreshes the bank datasets
- removes uploads older than a day
- withdraws open decisions older than a day

## Before you start

You need:
- a GitHub account
- a Vercel account. Sign in to Vercel with GitHub, on the free Hobby plan.
- your Supabase project

## Step 1. Make a separate schema for the showcase

Visitors will use the showcase, so keep it apart from the data you work with. A schema is a separate set of tables in the same database.

1. Copy the whole haara-v4 folder to a new folder, for example `haara-showcase`. Your own copy and its config.env stay untouched.
2. Open PowerShell in the new folder and run:

   ```
   python setup.py
   ```

   - Step 1 of setup checks the connection string. Press Enter to keep the one you have.
   - Step 2 asks for the schema. Type `showcase`. The demo bank goes there.
   - Step 3 builds the bank and trains the models. This takes one to three minutes.
   - Step 4 creates your administrator account. When it asks "Create a demonstration team with one person in each role?", answer Yes. The demo buttons sign visitors in as these people.

3. Open config.env in the new folder. You need these values in Step 3: DATABASE_URL, HAARA_SCHEMA and COMPANY_NAME.

Use the Session pooler connection string (host ending pooler.supabase.com, port 5432). Do not use the Transaction pooler on port 6543. Analytixa selects its schema once per connection, and the transaction pooler does not keep that setting.

## Step 2. Put the code on GitHub

This gives Vercel a copy of the code to build from. Run these in the haara-showcase folder:

1. `git init`. This starts tracking the folder with Git.
2. `git status`. This lists the files Git will include. Check that config.env is not listed. The .gitignore file keeps it out.
3. `git add .` and then `git commit -m "Analytixa showcase"`. This records the files in a first commit.
4. On github.com, create a new private repository named analytixa-showcase. Do not add a README.
5. Connect the folder to that repository and upload the code, using the two lines GitHub shows under "push an existing repository":

   ```
   git remote add origin https://github.com/YOUR-NAME/analytixa-showcase.git
   git push -u origin main
   ```

   If Git says the branch is called master, run `git branch -M main` first.

## Step 3. Create the Vercel project

1. On vercel.com, click Add New, then Project, and import analytixa-showcase. Vercel reads requirements.txt and server.py and sets up FastAPI on its own.
2. Before you click Deploy, open Environment Variables and add these:

   | Name | Value | What it does |
   |---|---|---|
   | DATABASE_URL | from config.env | Connects to Supabase |
   | HAARA_SCHEMA | showcase | Uses the showcase tables |
   | COMPANY_NAME | from config.env | The bank name on screen |
   | HAARA_SHOWCASE | 1 | Turns on the demo buttons and read-only settings |
   | CRON_SECRET | any long random text | Lets only Vercel run the daily job |
   | OPENROUTER_API_KEY | optional | Lets the language model read questions. Leave it out to use the built-in rules only. |

   To make a random CRON_SECRET, run this in PowerShell:
   ```
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

3. Click Deploy. The build takes two to four minutes, because it installs pandas, scikit-learn and scipy.

## Step 4. Check it

1. Open the link Vercel shows, for example analytixa-showcase.vercel.app. The first load takes a few seconds while the server starts.
2. Click Analyst under Try the demo. Ask "What is the delinquency rate by branch in Lagos?" and wait for Done.
3. Open Your data, open Loan portfolio under Bank data, and click Create a dashboard.
4. In Vercel, open the project, then Settings, then Cron Jobs. One job, /api/cron/daily, should be listed. Click Run to test it.

## Updating the showcase

Copy the changed files into haara-showcase, then run:

```
git add .
git commit -m "Update"
git push
```

Vercel builds and publishes the new version on its own.

## If something goes wrong

- **"This deployment has no DATABASE_URL"**: the environment variable is missing. Add it in Settings, Environment Variables, then redeploy.
- **"This database has no bank yet"**: Step 1 did not finish for the schema in HAARA_SCHEMA. Run setup.py again in the haara-showcase folder.
- **Pages are slow or fail after a week of no visitors**: Supabase paused the project. Open it in the Supabase dashboard and click Restore. The daily job normally prevents this.
- **Errors**: open the project in Vercel and click Logs. Error references such as E-1A2B3C appear there with the full details. Passwords and keys are removed before anything is logged.

Do not run the test suites against the showcase schema. They create test tasks under the demo accounts.
