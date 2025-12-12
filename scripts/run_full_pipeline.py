import sys
import subprocess

def run_cmd(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")

def main():
    if len(sys.argv) != 3:
        print("Usage: swiftsvd-run-pipeline --config <config.yaml>")
        sys.exit(1)

    config_arg = sys.argv[-1]
    run_cmd(f"swiftsvd-gather-activations --config {config_arg}")
    run_cmd(f"swiftsvd-profile-layers --config {config_arg}")
    run_cmd(f"swiftsvd-compress --config {config_arg}")
    run_cmd(f"swiftsvd-evaluate --config {config_arg}")
    print("✅ Full pipeline completed!")

if __name__ == "__main__":
    main()