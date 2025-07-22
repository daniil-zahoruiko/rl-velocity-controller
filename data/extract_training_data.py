import argparse
import json
import os

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message

from at_messages.msg import DynamicsState, TargetThrust


class RecordedTopic:
    def __init__(self, name, type, output, data_func):
        self.name = name
        self.type = type
        self.output = output
        self.data_func = data_func
        self.recorded_data = []

    def save_data(self):
        with open(self.output, "w") as f:
            f.write("[\n")
            for d in self.recorded_data:
                f.write(json.dumps(d))
                if d == self.recorded_data[-1]:
                    f.write("\n")
                    break
                f.write(",\n")

            f.write("]")


STATE_TOPIC = RecordedTopic(
    "/dynamics/state",
    DynamicsState,
    "states.json",
    lambda x: {"velocity": replace_nans(x.velocity.tolist())},
)
TARGET_STATE_TOPIC = RecordedTopic(
    "/dynamics/target",
    DynamicsState,
    "target_states.json",
    lambda x: {
        "velocity": replace_nans(x.velocity.tolist()),
        "position": replace_nans(x.position.tolist()),
    },
)
TARGET_THRUST_TOPIC = RecordedTopic(
    "/thrusters/target_thrust",
    TargetThrust,
    "actions.json",
    lambda x: {"thrust": replace_nans(x.target_thrust.tolist())},
)

TOPICS = [STATE_TOPIC, TARGET_STATE_TOPIC, TARGET_THRUST_TOPIC]


def create_files():
    for topic in TOPICS:
        if os.path.exists(topic.output):
            os.remove(topic.output)
        os.mknod(topic.output)


def replace_nans(items):
    return [None if np.isnan(item) else item for item in items]


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

    topic_mapping = dict([(topic.name, topic) for topic in TOPICS])
    while reader.has_next():
        topic_name, data, timestamp = reader.read_next()

        if topic_name not in topic_mapping.keys():
            continue

        msg = deserialize_message(data, topic_mapping[topic_name].type)
        topic_mapping[topic_name].recorded_data.append(
            {"timestamp": timestamp, "data": topic_mapping[topic_name].data_func(msg)}
        )

    create_files()
    for topic in TOPICS:
        topic.save_data()


if __name__ == "__main__":
    main()
