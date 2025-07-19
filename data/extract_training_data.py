import argparse
import os

import rosbag2_py
from rclpy.serialization import deserialize_message

from at_messages.msg import DynamicsState, TargetThrust

STATE_TOPIC = "/dynamics/state"
TARGET_STATE_TOPIC = "/dynamics/target"
TARGET_THRUST_TOPIC = "/thrusters/target_thrust"
TOPIC_NAMES = [STATE_TOPIC, TARGET_STATE_TOPIC, TARGET_THRUST_TOPIC]
TOPIC_TYPES = {
    STATE_TOPIC: DynamicsState,
    TARGET_STATE_TOPIC: DynamicsState,
    TARGET_THRUST_TOPIC: TargetThrust,
}
TOPIC_OUTPUTS = {
    STATE_TOPIC: "states.json",
    TARGET_STATE_TOPIC: "target_states.json",
    TARGET_THRUST_TOPIC: "actions.json",
}


def create_files():
    for file in TOPIC_OUTPUTS.values():
        if os.path.exists(file):
            os.remove(file)
        os.mknod(file)


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

    create_files()

    # cnt to limit messages for testing
    cnt = 0
    while reader.has_next() and cnt <= 30:
        topic_name, data, timestamp = reader.read_next()

        if topic_name not in TOPIC_NAMES:
            continue

        msg = deserialize_message(data, TOPIC_TYPES[topic_name])
        cnt += 1


if __name__ == "__main__":
    main()
