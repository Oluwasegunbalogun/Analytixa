"""First-time setup for Analytixa HAARA.

Connects the app to your Postgres database (a Supabase project or any Postgres
14+ server), builds the fictional bank and creates the staff accounts you want.

Interactive:      python setup.py
Non-interactive:  python setup.py --database-url URL --company "Lagoon Trust Bank" \
                      --admin-name "Ada Obi" --admin-email ada@bank.test --admin-password secret123 \
                      --demo-team --demo-password secret123
Add one user:     python setup.py --add-user
"""
import argparse
import getpass
import os
import re
import sys

from app import config

DEMO_TEAM = [
    ("Tola Adeyemi", "analyst"),
    ("Chidi Okafor", "loan_officer"),
    ("Hauwa Bello", "credit_manager"),
    ("Kunle Lawal", "compliance_officer"),
    ("Ngozi Eze", "retail_manager"),
    ("Musa Garba", "branch_ops"),
]


def ask(prompt, default=None, secret=False, check=None):
    while True:
        shown = f"{prompt} [{default}]: " if default else f"{prompt}: "
        v = getpass.getpass(shown) if secret else input(shown)
        v = v.strip() or (default or "")
        err = check(v) if check else (None if v else "This is required.")
        if not err:
            return v
        print(f"  {err}")


def ask_yes(prompt, default=True):
    v = input(f"{prompt} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    return default if not v else v.startswith("y")


def ask_password(label="Password"):
    while True:
        p1 = ask(f"{label} (at least 8 characters)", secret=True,
                 check=lambda v: None if len(v) >= 8 else "Use at least 8 characters.")
        p2 = getpass.getpass("Type it again: ")
        if p1 == p2:
            return p1
        print("  The two entries differ. Try again.")


def _email_check(v):
    return None if re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", v) else "Enter a valid email address."


def slug(name):
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return (s.split("_")[0] or "bank")[:30]


def domain_for(company):
    return re.sub(r"[^a-z0-9]", "", company.lower())[:20] + ".test"


def test_connection(url):
    import psycopg
    try:
        with psycopg.connect(url, connect_timeout=15, prepare_threshold=None) as c:
            v = c.execute("SHOW server_version").fetchone()[0]
        return None, v
    except Exception as e:
        return str(e).strip().splitlines()[0], None


def pick_role(auth):
    print("  Roles:")
    for i, (r, t, d) in enumerate(auth.ROLES, 1):
        print(f"   {i}. {t}: {d}")
    n = ask("  Role number", check=lambda v: None if v.isdigit() and 1 <= int(v) <= len(auth.ROLES)
            else f"Enter a number from 1 to {len(auth.ROLES)}.")
    return auth.ROLES[int(n) - 1][0]


def add_users_loop(auth):
    while ask_yes("Add a staff user?", default=False):
        name = ask("  Full name")
        email = ask("  Email", check=_email_check)
        role = pick_role(auth)
        pw = ask_password("  Password")
        try:
            u = auth.create_user(name, email, role, pw)
            print(f"  Created {u['full_name']} as {u['role_title']}.")
        except ValueError as e:
            print(f"  {e}")


def build(progress=print):
    from app import auth
    import server
    auth.init()
    server.boot(verbose=True)


def main():
    ap = argparse.ArgumentParser(description="Set up Analytixa HAARA.")
    ap.add_argument("--database-url")
    ap.add_argument("--company")
    ap.add_argument("--schema")
    ap.add_argument("--admin-name")
    ap.add_argument("--admin-email")
    ap.add_argument("--admin-password")
    ap.add_argument("--demo-team", action="store_true")
    ap.add_argument("--demo-password")
    ap.add_argument("--add-user", action="store_true", help="only add staff users to an existing setup")
    a = ap.parse_args()
    interactive = not (a.database_url and a.admin_email and a.admin_password)
    existing = config.read_file()

    if a.add_user:
        if not config.configured():
            sys.exit("Run python setup.py first.")
        from app import auth
        auth.init()
        add_users_loop(auth)
        return

    print("\nAnalytixa HAARA setup\n")
    print("Step 1 of 4. Database")
    print("In Supabase open your project, choose Connect, and copy the Session pooler")
    print("connection string. Replace [YOUR-PASSWORD] with your database password.\n")
    url = a.database_url
    while True:
        if not url:
            url = ask("Connection string", default=existing.get("DATABASE_URL") and "keep current")
            if url == "keep current":
                url = existing["DATABASE_URL"]
        url = url.strip().strip('"')
        if "[YOUR-PASSWORD]" in url:
            print("  Replace [YOUR-PASSWORD] with your real database password.")
            url = None
            continue
        if "supabase" in url and "sslmode=" not in url:
            url += ("&" if "?" in url else "?") + "sslmode=require"
        err, ver = test_connection(url)
        if not err:
            print(f"  Connected. Postgres {ver}.")
            break
        print(f"  Could not connect: {err}")
        if not interactive:
            sys.exit(1)
        url = None

    print("\nStep 2 of 4. Company")
    company = a.company or (ask("Bank name", default=existing.get("COMPANY_NAME", "Lagoon Trust Bank"))
                            if interactive else existing.get("COMPANY_NAME", "Lagoon Trust Bank"))
    schema = a.schema or existing.get("HAARA_SCHEMA") or slug(company)
    if interactive and not a.schema:
        schema = ask("Database schema for this bank", default=schema,
                     check=lambda v: None if re.fullmatch(r"[a-z_][a-z0-9_]{0,40}", v)
                     else "Use lower case letters, digits and underscores.")
    values = dict(existing, DATABASE_URL=url, COMPANY_NAME=company, HAARA_SCHEMA=schema)
    config.write_file(values)
    for k, v in values.items():
        os.environ[k] = v
    print(f"  Saved to config.env. Data goes in schema '{schema}'.")

    print("\nStep 3 of 4. Building the bank")
    print("All customers, accounts and transactions are synthetic.")
    try:
        build()
    except Exception as e:
        import psycopg
        if not isinstance(e, psycopg.OperationalError):
            raise
        print("\n  The database stopped answering while the bank was being built.")
        print("  " + str(e).splitlines()[0])
        print("  Nothing is lost. Run  python setup.py  again and press Enter to keep the current settings.")
        print("  It continues from where it stopped.")
        sys.exit(1)

    from app import auth
    from app import l1_data as db
    print("\nStep 4 of 4. Staff accounts")
    have_admin = db.one("SELECT full_name, email FROM app_users WHERE role='admin' AND active")
    if have_admin and interactive:
        print(f"  Administrator already set up: {have_admin['full_name']} <{have_admin['email']}>.")
    elif not have_admin:
        print("Create the administrator. This person manages staff, policies and settings.")
        name = a.admin_name or ask("  Full name")
        email = a.admin_email or ask("  Email", check=_email_check)
        pw = a.admin_password or ask_password("  Password")
        u = auth.create_user(name, email, "admin", pw)
        print(f"  Created {u['full_name']} as Administrator.")

    demo = a.demo_team or (interactive and ask_yes(
        "\nCreate a demonstration team with one person in each role?", default=True))
    if demo:
        pw = a.demo_password or ask_password("  One password for the whole team")
        dom = domain_for(company)
        for name, role in DEMO_TEAM:
            email = name.split()[0].lower() + "@" + dom
            try:
                auth.create_user(name, email, role, pw)
                print(f"  {auth.ROLE_TITLES[role]:<20} {name:<15} {email}")
            except ValueError as e:
                print(f"  {e}")
    if interactive:
        add_users_loop(auth)

    print("\nSetup is complete. Start the app with start.bat (Windows) or ./start.sh,")
    print("or run: python server.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSetup stopped. Run python setup.py to start again.")
