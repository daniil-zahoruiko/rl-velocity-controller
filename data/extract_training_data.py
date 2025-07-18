import argparse

import rosbag2_py

TOPIC_NAMES = ["/dynamics/state", "/dynamics/target", "/thrusters/target_thrust"]


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("bag_path", nargs=1)

    args = argparser.parse_args()
    bag_path = args.bag_path[0]

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(
            uri=bag_path,
            storage_id="sqlite3",
        ),
        rosbag2_py.ConverterOptions("", "cdr"),
    )

    topic_types = reader.get_all_topics_and_types()
    type_map = {topic.name: topic.type for topic in topic_types}
    print(type_map)


if __name__ == "__main__":
    main()