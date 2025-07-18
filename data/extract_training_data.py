import sqlite3
import argparse

TOPIC_NAMES = ['/dynamics/state', '/dynamics/target', '/thrusters/target_thrust']

def get_topic_id_mapping(cursor: sqlite3.Cursor):
    res = cursor.execute('SELECT id, name FROM topics WHERE name IN (%s)' % ','.join('?' * len(TOPIC_NAMES)), TOPIC_NAMES)
    id_mapping = {}
    for (id, name) in res.fetchall():
        id_mapping[name] = id

    return id_mapping

def get_messages(cursor: sqlite3.Cursor, topic_id):
    res = cursor.execute('SELECT timestamp, data FROM messages WHERE topic_id=? LIMIT 10', (topic_id, )) # NOTE: LIMIT only for testing purposes
    return res.fetchall()

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('db_path', nargs=1)

    args = argparser.parse_args()

    db_path = args.db_path[0]
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    id_mapping = get_topic_id_mapping(cur)
    states = get_messages(cur, id_mapping['/dynamics/state'])
    targets = get_messages(cur, id_mapping['/dynamics/target'])
    actions = get_messages(cur, id_mapping['/thrusters/target_thrust'])
    print(states)
    print(targets)
    print(actions)

if __name__ == '__main__':
    main()