import json
import os
import socket
import sys

import hydra
import numpy as np
from omegaconf import DictConfig

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from memory.hmsg.graph.graph import Graph

# pylint: disable=all


def send_goal_via_udp(target_name, map_coords, port=5005):
    """Sends the target coordinates via a local UDP socket."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    data = {
        "name": target_name,
        "x": float(map_coords[0]),
        "y": float(map_coords[1]),
        "z": float(map_coords[2]),
    }
    sock.sendto(json.dumps(data).encode("utf-8"), ("127.0.0.1", port))
    print(f"\n[UDP] Sent Top-1 Goal ({target_name}) to port {port}")


@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="visualize_query_graph_icra_demo_dat",
)
def main(params: DictConfig):
    T_switch_axis = np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64
    )
    T_tomap = np.linalg.inv(T_switch_axis)

    hmsg = Graph(params)
    print("\n" + "=" * 50)
    print(f"Loading scene graph from: {params.main.graph_path}")
    hmsg.load_hmsg_graph(params.main.graph_path)
    print("Scene graph loaded successfully!")
    print("=" * 50)

    negative_labels = ["background", "wall", "ceiling", "floor", "door"]
    all_room_indices = list(range(len(hmsg.rooms)))

    try:
        while True:
            query_label = input(
                "\nEnter the object you want to find (or type 'q' to quit): "
            ).strip()
            if query_label.lower() in ["q", "quit", "exit"]:
                break
            if not query_label:
                continue

            print(f"\nSearching for '{query_label}' using local CLIP matching...")

            object_ids, room_ids, object_scores = hmsg.query_hmsg_object(
                query=query_label,
                floor_id=-1,
                room_ids=all_room_indices,
                query_method="clip",
                top_k=5,
                negative_prompt=negative_labels,
            )

            if not object_ids:
                print(f"  -> No objects matched '{query_label}'.")
                continue

            print(f"\n--- Top Matches for '{query_label}' ---")
            for rank, (obj_idx, room_idx, score) in enumerate(
                zip(object_ids, room_ids, object_scores), 1
            ):
                obj = hmsg.objects[obj_idx]
                room = hmsg.rooms[room_idx]

                obj_center_raw = obj.pcd.get_center()
                obj_center_h = np.hstack((obj_center_raw, 1.0))
                obj_center_in_map = (T_tomap @ obj_center_h)[:3]

                print(f"#{rank} Detected Label: **{obj.name}**")
                print(f"    Confidence Score: {score:.4f}")
                print(
                    f"    Map Coordinates:  x={obj_center_in_map[0]:.3f}, y={obj_center_in_map[1]:.3f}, z={obj_center_in_map[2]:.3f}\n"
                )

            # --- Send Top 1 to ROS via UDP ---
            top1_obj = hmsg.objects[object_ids[0]]
            top1_center_h = np.hstack((top1_obj.pcd.get_center(), 1.0))
            top1_center_in_map = (T_tomap @ top1_center_h)[:3]

            send_goal_via_udp(top1_obj.name, top1_center_in_map)

    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")


if __name__ == "__main__":
    main()
