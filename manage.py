"""Maintenance commands.

python manage.py verify      check the provenance ledger hash chain
python manage.py users       list staff accounts
python manage.py adduser     add a staff account (asks for the details)
python manage.py password EMAIL   set a new password for a staff account
python manage.py reset       rebuild the bank, policies, models and ledger (staff accounts are kept)
"""
import getpass
import sys

from app import config


def main():
    if not config.configured():
        sys.exit("Run python setup.py first.")
    from app import auth
    from app import l1_data as db
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "verify":
        print(db.verify_ledger())
    elif cmd == "users":
        for u in auth.list_users():
            print(f"{u['role_title']:<20} {u['full_name']:<24} {u['email']:<36} {'active' if u['active'] else 'disabled'}")
    elif cmd == "adduser":
        import setup
        auth.init()
        setup.add_users_loop(auth)
    elif cmd == "password" and len(sys.argv) > 2:
        u = db.one("SELECT user_id FROM app_users WHERE lower(email)=lower(?)", (sys.argv[2],))
        if not u:
            sys.exit("No user with that email.")
        pw = getpass.getpass("New password (at least 8 characters): ")
        auth.update_user(u["user_id"], "system:manage.py", password=pw)
        print("Password changed. Existing sessions for this user were signed out.")
    elif cmd == "reset":
        if input("This deletes all runs, decisions, policy versions and the ledger. Type RESET to continue: ") != "RESET":
            sys.exit("Nothing changed.")
        users = db.query("SELECT full_name, email, role, password_hash, active FROM app_users")
        db.drop_schema()
        import server
        server.boot(verbose=True)
        for u in users:
            db.execute("INSERT INTO app_users (full_name, email, role, password_hash, active) VALUES (?,?,?,?,?)",
                       (u["full_name"], u["email"], u["role"], u["password_hash"], u["active"]))
        print(f"Rebuilt. {len(users)} staff accounts kept.")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
