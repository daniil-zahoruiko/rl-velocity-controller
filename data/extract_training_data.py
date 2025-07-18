import sqlite3
import argparse

TOPIC_NAMES = ['/dynamics/state', '/dynamics/target', '/thrusters/target_thrust']

def get_topic_id_mapping(cursor: sqlite3.Cursor):
    res = cursor.execute('SELECT id, name FROM topics WHERE name IN (%s)' % ','.join('?' * len(TOPIC_NAMES)), TOPIC_NAMES)
    id_mapping = {}
    for (id, name) in res.fetchall():
        id_mapping[name] = id

    return id_mapping

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('db_path', nargs=1)

    args = argparser.parse_args()

    db_path = args.db_path[0]
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    print(get_topic_id_mapping(cur))

if __name__ == '__main__':
    main()