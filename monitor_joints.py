import time
from lerobot.robots.so_follower import SO101Follower
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig

def main():
    print(" Connecting to robot and disabling torque to allow manual movement...")
    config = SO101FollowerConfig(port="/dev/ttyACM0", use_degrees=True, num_read_retries=5)
    robot = SO101Follower(config)
    robot.connect()
    
    # Disable torque so the user can easily move the arm by hand
    robot.bus.disable_torque()
    
    print("\n Torque disabled! You can now move the arm joints by hand.")
    print("Adjust the arm so the joints match the targets:")
    print("  Target: [shoulder_pan ~ -5, shoulder_lift ~ -99, elbow_flex ~ 99, wrist_flex ~ 0, wrist_roll ~ 0]")
    print("\nPress Ctrl+C to exit.\n")
    
    joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    
    try:
        while True:
            obs = robot.get_observation()
            state = [obs[f"{j}.pos"] for j in joints]
            print(
                f"\rCurrent: "
                f"pan: {state[0]:+6.1f} | "
                f"lift: {state[1]:+6.1f} | "
                f"elbow: {state[2]:+6.1f} | "
                f"wrist_flex: {state[3]:+6.1f} | "
                f"wrist_roll: {state[4]:+6.1f} | "
                f"gripper: {state[5]:+6.1f}",
                end="",
                flush=True
            )
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\n Re-enabling torque and disconnecting...")
    finally:
        robot.disconnect()

if __name__ == "__main__":
    main()
