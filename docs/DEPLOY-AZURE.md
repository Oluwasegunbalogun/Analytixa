# Deploying Analytixa to Azure with Azure for Students

This puts Analytixa on a public link at https://YOUR-APP-NAME.azurewebsites.net. Visitors pick a role under "Try the demo" and look around without a password. All data is synthetic.

## What you are building

| Part | Azure service | Cost on Azure for Students |
|---|---|---|
| The app | App Service, Linux, Python 3.12, Basic B1 plan | About $13 a month from your $100 credit, so about 7 months |
| The database | Azure Database for PostgreSQL, Flexible Server, Burstable B1ms, 32 GB | Free for 12 months (750 hours a month) |

App Service runs Analytixa as one long-running server, the same way it runs on your laptop. Background tasks, approvals and the hourly maintenance check work without changes. With Always On, the app never sleeps.

To spend nothing on the app, use the free F1 plan instead of B1. F1 sleeps after about 20 minutes without visitors, takes 30 to 60 seconds to wake, and allows 60 CPU minutes a day. You can move between F1 and B1 later with one command (see "Saving credit").

Showcase mode works as on Vercel:
- "Try the demo" buttons, one per role, with no password needed
- a banner saying the data is synthetic
- settings, policies and staff accounts are read-only
- a daily clean-up of uploads and open decisions older than a day

On Azure the clean-up runs inside the app, once a day. No scheduler is needed.

## Before you start

You need:
- your Azure for Students subscription, activated at azure.microsoft.com/free/students with your school email
- Python on your computer, the one you already use for Analytixa
- the Azure command line tool. Install it in PowerShell:

```
winget install -e --id Microsoft.AzureCLI
```

This installs the `az` command. Close PowerShell and open it again so the command is found.

Pick three names and use them in every step below:
- a resource group: `analytixa-rg`. It is a folder in Azure that holds everything for this project.
- a database server name: for example `analytixa-db-shegun`. It must be unique across Azure.
- an app name: for example `analytixa-shegun`. It becomes the web address and must be unique across Azure.

Pick one region for both, close to Lagos: `southafricanorth`. Student subscriptions allow only some regions. If a step fails with "RequestDisallowedByAzure" or a quota message, use `westeurope` or `northeurope` for both the database and the app.

## Step 1. Sign in to Azure

```
az login
```

This opens your browser. Sign in with the account that holds Azure for Students.

```
az account show --output table
```

This shows the active subscription. Check the name says Azure for Students. If you have several subscriptions, select it with `az account set --subscription "Azure for Students"`.

## Step 2. Create the resource group

```
az group create --name analytixa-rg --location southafricanorth
```

This creates the folder that holds the database and the app. To remove everything later, delete this one group.

## Step 3. Create the database

Choose a database password of at least 12 characters, using capital letters, small letters and digits. Leave out symbols, so the connection string needs no escaping.

```
az postgres flexible-server create --resource-group analytixa-rg --name analytixa-db-shegun --location southafricanorth --tier Burstable --sku-name Standard_B1ms --storage-size 32 --version 16 --admin-user analytixa --admin-password YourPassword123 --public-access None
```

This creates a PostgreSQL 16 server on the free B1ms size with 32 GB of storage. `--public-access None` starts it closed to everyone. The next two commands open it to the right places. It takes five to ten minutes.

```
az postgres flexible-server firewall-rule create --resource-group analytixa-rg --name analytixa-db-shegun --rule-name AllowAzureServices --start-ip-address 0.0.0.0 --end-ip-address 0.0.0.0
```

This lets services inside Azure, such as your app, reach the database. The 0.0.0.0 rule is Azure's way of saying "Azure services only".

```
az postgres flexible-server firewall-rule create --resource-group analytixa-rg --name analytixa-db-shegun --rule-name MyComputer --start-ip-address YOUR.PUBLIC.IP --end-ip-address YOUR.PUBLIC.IP
```

This lets your own computer reach the database, so setup.py can build the bank in Step 4. Find your public IP by searching "what is my IP" in a browser. If your internet provider changes your IP, run this command again with the new one.

Your connection string is:

```
postgresql://analytixa:YourPassword123@analytixa-db-shegun.postgres.database.azure.com:5432/postgres?sslmode=require
```

`sslmode=require` makes the connection encrypted, which Azure demands.

## Step 4. Build the bank in the Azure database

Keep your working copy apart from the showcase.

1. Copy the whole haara-v4 folder to a new folder, `haara-azure`. Your own config.env stays untouched.
2. Open PowerShell in haara-azure and run:

```
python setup.py
```

- Step 1 of setup asks for the connection string. Paste the one from Step 3.
- Step 2 asks for the bank name and the schema. Press Enter to keep the defaults.
- Step 3 builds the bank and trains the models. This takes two to five minutes over the internet.
- Step 4 creates your administrator account. When it asks "Create a demonstration team with one person in each role?", answer Yes. The demo buttons sign visitors in as these people.
- The demo buttons only use accounts whose email ends in .test, so your own administrator account is never offered to visitors. To give visitors an Administrator button too, run `python setup.py --add-user` and add an Administrator with an address such as admin@lagoontrustbank.test.

3. Open config.env in haara-azure. You need DATABASE_URL, HAARA_SCHEMA and COMPANY_NAME in Step 6.

## Step 5. Create the app

```
az appservice plan create --name analytixa-plan --resource-group analytixa-rg --location southafricanorth --is-linux --sku B1
```

This creates the Linux server the app runs on. Use `--sku F1` instead of B1 for the free plan.

```
az webapp create --resource-group analytixa-rg --plan analytixa-plan --name analytixa-shegun --runtime "PYTHON:3.12"
```

This creates the web app with Python 3.12 and gives it the address https://analytixa-shegun.azurewebsites.net.

## Step 6. Give the app its settings

```
az webapp config appsettings set --resource-group analytixa-rg --name analytixa-shegun --settings DATABASE_URL="postgresql://analytixa:YourPassword123@analytixa-db-shegun.postgres.database.azure.com:5432/postgres?sslmode=require" HAARA_SCHEMA=lagoon COMPANY_NAME="Lagoon Trust Bank" HAARA_SHOWCASE=1 SCM_DO_BUILD_DURING_DEPLOYMENT=true
```

This stores the settings the app reads at start-up. Use the values from your config.env:
- DATABASE_URL connects to the database.
- HAARA_SCHEMA and COMPANY_NAME point at the bank setup.py built.
- HAARA_SHOWCASE=1 turns on the demo buttons and the read-only settings.
- SCM_DO_BUILD_DURING_DEPLOYMENT=true tells Azure to install the packages in requirements.txt when you upload the code.

To let the language model read questions, add `OPENROUTER_API_KEY=your-key` to the same command. Without it, Analytixa uses its built-in rules. Settings in Azure stay on the server, and the page never shows them.

```
az webapp config set --resource-group analytixa-rg --name analytixa-shegun --startup-file "python -m uvicorn server:app --host 0.0.0.0 --port 8000" --always-on true
```

This tells Azure how to start Analytixa: one uvicorn server on port 8000, the port App Service sends traffic to. `--always-on true` keeps it running between visits. On the free F1 plan, leave out `--always-on true`, because F1 does not support it.

```
az webapp update --resource-group analytixa-rg --name analytixa-shegun --https-only true
```

This sends every visitor to the encrypted https address.

## Step 7. Upload the code

In PowerShell, in the haara-azure folder:

```
python make_azure_zip.py
```

This builds analytixa-azure.zip in the folder above. The zip holds the app without config.env, logs or tests, so your passwords never leave your computer.

```
az webapp deploy --resource-group analytixa-rg --name analytixa-shegun --src-path ..\analytixa-azure.zip --type zip
```

This uploads the zip. Azure installs the packages and starts the app, which takes three to six minutes the first time.

## Step 8. Check it

1. Open https://analytixa-shegun.azurewebsites.net. The first load after an upload takes a minute while the app starts.
2. Click Analyst under Try the demo. Ask "What is the delinquency rate by branch in Lagos?" and wait for Done.
3. Open Your data, open Loan portfolio under Bank data, and ask "Create a dashboard".

To watch the app's log live while you test:

```
az webapp log config --resource-group analytixa-rg --name analytixa-shegun --docker-container-logging filesystem
az webapp log tail --resource-group analytixa-rg --name analytixa-shegun
```

The first command turns on log collection. The second streams the log to PowerShell. Press Ctrl+C to stop. Error references such as E-1A2B3C appear here with the details. Passwords and keys are removed before anything is logged. The same log is kept on the server under /home/LogFiles/analytixa.

## Updating the showcase

Copy the changed files into haara-azure, then run Step 7 again. The settings from Step 6 stay in place.

## Saving credit

- Check your remaining credit at microsoftazuresponsorships.com.
- To move to the free plan: `az appservice plan update --name analytixa-plan --resource-group analytixa-rg --sku F1`. Turn Always On off first with `az webapp config set --resource-group analytixa-rg --name analytixa-shegun --always-on false`.
- To move back: `az appservice plan update --name analytixa-plan --resource-group analytixa-rg --sku B1`, then turn Always On back on.
- Stopping the app does not stop the plan's charge. Only a smaller plan or deleting the plan does.
- To remove everything: `az group delete --name analytixa-rg`. This deletes the app, the plan and the database together.

When the credit runs out, Azure shuts the resources down instead of charging you. Renew Azure for Students each year while you are a student.

## If something goes wrong

- **Application Error page**: open the log (Step 8). A message about DATABASE_URL means a setting is missing or mistyped.
- **"This database has no bank yet"**: Step 4 did not finish. Run setup.py again in haara-azure.
- **Could not reach the database**: check the AllowAzureServices rule from Step 3, and that the connection string ends with ?sslmode=require.
- **setup.py cannot connect from your computer**: your public IP changed. Run the MyComputer firewall command again with the new IP.
- **The page is slow on F1**: the app was asleep. Move to B1, or open the link a minute before you need it.

Do not run the test suites against the Azure database. They create test tasks under the demo accounts.
