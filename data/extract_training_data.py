import sqlite3
import argparse

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('db_path', nargs=1)

    args = argparser.parse_args()

    db_path = args.db_path[0]
    con = sqlite3.connect(db_path)
    cur = con.cursor()

if __name__ == '__main__':
    main()