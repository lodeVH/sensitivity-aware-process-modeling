import argparse
import csv
import re
from pathlib import Path


CSV_DIR = Path(__file__).resolve().parent
DT_PATTERN = re.compile(r"_dt([0-9eE+\-.]+)(?=(_|\.csv$))", re.IGNORECASE)


def list_csv_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.csv"), key=lambda path: path.name.lower())


def extract_dt_seconds(file_path: Path) -> int:
    match = DT_PATTERN.search(file_path.name)
    if not match:
        raise ValueError(
            f"Could not find sample time in filename: {file_path.name}. "
            "Expected a pattern like '_dt100' or '_dt1e2'."
        )
    dt_value = float(match.group(1))
    if dt_value <= 0:
        raise ValueError(f"Extracted non-positive sample time from filename: {file_path.name}")
    if not dt_value.is_integer():
        raise ValueError(
            f"Sample time in filename must resolve to an integer number of seconds: {file_path.name}"
        )
    return int(dt_value)


def ask_for_file(csv_files: list[Path]) -> Path:
    print("Available CSV files:\n")
    for index, file_path in enumerate(csv_files, start=1):
        print(f"{index}. {file_path.name}")

    while True:
        choice = input("\nSelect a CSV by number: ").strip()
        if not choice.isdigit():
            print("Please enter a valid number.")
            continue

        selected_index = int(choice)
        if 1 <= selected_index <= len(csv_files):
            return csv_files[selected_index - 1]

        print(f"Please enter a number between 1 and {len(csv_files)}.")


def ask_for_new_dt(old_dt: int) -> int:
    while True:
        raw_value = input(
            f"Enter the new sample time in seconds (current is {old_dt}s): "
        ).strip()
        if not raw_value.isdigit():
            print("Please enter a positive integer.")
            continue

        new_dt = int(raw_value)
        if new_dt < old_dt:
            print("The new sample time must be greater than or equal to the current one.")
            continue

        if new_dt % old_dt != 0:
            print(
                f"The new sample time must be an integer multiple of {old_dt}s."
            )
            continue

        return new_dt


def build_output_path(input_path: Path, new_dt: int) -> Path:
    new_name = DT_PATTERN.sub(f"_dt{new_dt}", input_path.name, count=1)
    return input_path.with_name(new_name)


def downsample_csv(input_path: Path, output_path: Path, keep_every_nth_row: int) -> tuple[int, int]:
    kept_rows = 0
    skipped_rows = 0

    with input_path.open("r", newline="", encoding="utf-8") as infile, output_path.open(
        "w", newline="", encoding="utf-8"
    ) as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        header = next(reader)
        writer.writerow(header)

        for row_index, row in enumerate(reader):
            if row_index % keep_every_nth_row == 0:
                writer.writerow(row)
                kept_rows += 1
            else:
                skipped_rows += 1

    return kept_rows, skipped_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Downsample a CSV so the effective dt in the filename and the retained rows match a coarser sample time."
    )
    parser.add_argument("--input", type=str, default="", help="Optional input CSV path.")
    parser.add_argument("--new-dt", type=int, default=0, help="Optional target sample time in seconds.")
    parser.add_argument("--output", type=str, default="", help="Optional explicit output CSV path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.input:
        selected_file = Path(args.input).expanduser().resolve()
        if not selected_file.exists():
            raise FileNotFoundError(f"Input CSV not found: {selected_file}")
    else:
        csv_files = list_csv_files(CSV_DIR)
        if not csv_files:
            print(f"No CSV files found in: {CSV_DIR}")
            return
        selected_file = ask_for_file(csv_files)

    old_dt = extract_dt_seconds(selected_file)
    new_dt = args.new_dt if args.new_dt else ask_for_new_dt(old_dt)
    if new_dt < old_dt:
        raise ValueError(f"The new sample time must be greater than or equal to the current one ({old_dt}s).")
    if new_dt % old_dt != 0:
        raise ValueError(f"The new sample time must be an integer multiple of {old_dt}s.")
    factor = new_dt // old_dt

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = build_output_path(selected_file, new_dt)

    if output_path.resolve() == selected_file.resolve():
        print(
            "\nThe output filename would be the same as the input filename. "
            "Choose a different new sample time or rename the existing file first."
        )
        return

    print(f"\nInput file:  {selected_file.name}")
    print(f"Current dt:  {old_dt}s")
    print(f"New dt:      {new_dt}s")
    print(f"Keep factor: 1 out of every {factor} rows")
    print(f"Output file: {output_path.name}\n")

    kept_rows, skipped_rows = downsample_csv(selected_file, output_path, factor)

    print("Finished.")
    print(f"Rows kept:    {kept_rows}")
    print(f"Rows removed: {skipped_rows}")
    print(f"Saved to:     {output_path}")


if __name__ == "__main__":
    main()
