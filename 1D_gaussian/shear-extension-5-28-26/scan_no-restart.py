import re
import shutil
import subprocess
from pathlib import Path

# ==================================================
# USER SETTINGS
# ==================================================

EXECUTABLE = "./resonance_superwide_beta_00"

DATA_DIR = Path("./data")
INPUT_FILE = DATA_DIR / "BOUT.inp"

# ----------------------------------
# T0 scan
# ----------------------------------

T0_VALUES = [
    0.0001,
    0.001,
    0.002,
    0.003,
    0.004,
    0.005,
    0.006,
    0.007,
    0.008,
    0.009
]

# ----------------------------------
# beta scan
# ----------------------------------

BETA_VALUES = [
    0.01,
    0.05,
    0.1,
    0.2,
    0.5,
    1.0,
    1.25
]


# ==================================================
# HELPERS
# ==================================================

def fmt_val(x):
    return f"{x:g}".replace(".", "p")


def replace_param(text, section, key, value):

    lines = text.splitlines()

    out = []

    in_section = False

    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")

    for line in lines:

        stripped = line.strip()

        if stripped.startswith("[") and stripped.endswith("]"):
            in_section = (stripped == f"[{section}]")

        if in_section and pattern.match(line):
            out.append(f"{key} = {value}")
        else:
            out.append(line)

    return "\n".join(out) + "\n"


def update_input_file(T0_value, beta_value):

    with open(INPUT_FILE, "r") as f:
        text = f.read()

    text = replace_param(text, "T", "A", T0_value)

    text = replace_param(text, "hw", "beta", beta_value)

    with open(INPUT_FILE, "w") as f:
        f.write(text)


def run_simulation():

    print(f"Running {EXECUTABLE}")

    proc = subprocess.run(
        [EXECUTABLE],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    return proc.returncode, proc.stdout


def save_output(beta_value, T0_value, status):

    tag = "OK" if status == 0 else "CRASH"

    beta_tag = fmt_val(beta_value)
    T0_tag = fmt_val(T0_value)

    out_dir = Path(f"beta_{beta_tag}") / f"T0_{T0_tag}_{tag}"

    if out_dir.exists():
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True)

    for item in DATA_DIR.iterdir():

        dest = out_dir / item.name

        if item.is_dir():
            shutil.copytree(item, dest)

        else:
            shutil.copy2(item, dest)


# ==================================================
# MAIN LOOP
# ==================================================

def main():

    for beta in BETA_VALUES:

        print(f"\n==============================")
        print(f"Running beta = {beta}")
        print(f"==============================")

        for T0 in T0_VALUES:

            print(f"\n=== Running T0 = {T0} ===")

            try:

                # ----------------------------------
                # Update input file
                # ----------------------------------

                update_input_file(T0, beta)

                # ----------------------------------
                # Run simulation
                # ----------------------------------

                code, output = run_simulation()

                if code != 0:

                    print(f"Run FAILED for T0 = {T0}")
                    print(output)

                # ----------------------------------
                # Save results
                # ----------------------------------

                save_output(beta, T0, code)

            except Exception as e:

                print(f"Unexpected error at T0 = {T0}: {e}")

                save_output(beta, T0, status=1)

                continue

    print("\nAll runs complete.")


if __name__ == "__main__":
    main()