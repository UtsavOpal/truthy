"""
╔══════════════════════════════════════════════════════════════╗
║          HALLUCINATION DETECTION FRAMEWORK v1.0              ║
║                  Powered by Claude AI                        ║
╚══════════════════════════════════════════════════════════════╝

Usage:
    python main.py                  → Interactive mode
    python main.py --demo           → Run all sample test cases
    python main.py --test 1         → Run a specific sample test (1-5)
    python main.py --input file.json → Detect from a JSON file
"""

import argparse
import sys

# Load .env file automatically (must come before importing detector)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv optional; user can export env vars manually

from src.detector import HallucinationDetector
from src.display import ResultDisplay
from src.samples import SAMPLE_TESTS
from src.models import DetectionInput


def run_interactive(detector: HallucinationDetector):
    """Interactive mode: user types paragraph, question, answer."""
    display = ResultDisplay()
    display.print_banner()

    while True:
        print("\n" + "─" * 65)
        print("  📝  NEW DETECTION  (type 'quit' to exit)")
        print("─" * 65)

        paragraph = _multiline_input(
            "\n[1/3] Context Paragraph\n"
            "      (press Enter twice to finish, or leave blank for world knowledge):\n> "
        )
        if paragraph.lower().strip() == "quit":
            break

        question = input("\n[2/3] Question:\n> ").strip()
        if question.lower() == "quit":
            break
        if not question:
            print("  ⚠  Question is required.")
            continue

        answer = _multiline_input("\n[3/3] Model's Answer\n      (press Enter twice to finish):\n> ")
        if answer.lower().strip() == "quit":
            break
        if not answer:
            print("  ⚠  Answer is required.")
            continue

        detection_input = DetectionInput(
            paragraph=paragraph,
            question=question,
            answer=answer
        )

        print("\n  ⏳  Analyzing...\n")
        result = detector.detect(detection_input)
        display.show_result(detection_input, result)

        again = input("\n  Run another? (y/n): ").strip().lower()
        if again != "y":
            break

    print("\n  👋  Goodbye!\n")


def run_demo(detector: HallucinationDetector):
    """Run all 5 built-in sample test cases."""
    display = ResultDisplay()
    display.print_banner()
    print(f"\n  Running {len(SAMPLE_TESTS)} sample test cases...\n")

    for i, sample in enumerate(SAMPLE_TESTS, 1):
        print(f"\n{'═' * 65}")
        print(f"  SAMPLE TEST #{i}")
        print(f"{'═' * 65}")
        detection_input = DetectionInput(**sample)
        print(f"\n  ⏳  Analyzing test #{i}...\n")
        result = detector.detect(detection_input)
        display.show_result(detection_input, result)

    print(f"\n{'═' * 65}")
    print("  ✅  All sample tests complete.")
    print(f"{'═' * 65}\n")


def run_single_test(detector: HallucinationDetector, test_num: int):
    """Run a specific sample test case by number."""
    display = ResultDisplay()
    display.print_banner()

    if test_num < 1 or test_num > len(SAMPLE_TESTS):
        print(f"\n  ⚠  Invalid test number. Choose 1–{len(SAMPLE_TESTS)}.\n")
        sys.exit(1)

    sample = SAMPLE_TESTS[test_num - 1]
    print(f"\n{'═' * 65}")
    print(f"  SAMPLE TEST #{test_num}")
    print(f"{'═' * 65}")
    detection_input = DetectionInput(**sample)
    print(f"\n  ⏳  Analyzing...\n")
    result = detector.detect(detection_input)
    display.show_result(detection_input, result)


def run_from_file(detector: HallucinationDetector, filepath: str):
    """Load input from a JSON file and run detection."""
    import json
    display = ResultDisplay()
    display.print_banner()

    try:
        with open(filepath) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"\n  ⚠  File not found: {filepath}\n")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"\n  ⚠  Invalid JSON: {e}\n")
        sys.exit(1)

    # Support single dict or list of dicts
    entries = [data] if isinstance(data, dict) else data
    for i, entry in enumerate(entries, 1):
        if len(entries) > 1:
            print(f"\n{'═' * 65}")
            print(f"  ENTRY #{i}")
            print(f"{'═' * 65}")
        detection_input = DetectionInput(
            paragraph=entry.get("paragraph", ""),
            question=entry["question"],
            answer=entry["answer"]
        )
        print(f"\n  ⏳  Analyzing...\n")
        result = detector.detect(detection_input)
        display.show_result(detection_input, result)


def _multiline_input(prompt: str) -> str:
    """Read multi-line input until double Enter."""
    print(prompt, end="", flush=True)
    lines = []
    while True:
        line = input()
        if line == "" and lines:
            break
        lines.append(line)
    return "\n".join(lines).strip()


def main():
    parser = argparse.ArgumentParser(
        description="Hallucination Detection Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--demo", action="store_true", help="Run all 5 sample test cases")
    parser.add_argument("--test", type=int, metavar="N", help="Run sample test N (1–5)")
    parser.add_argument("--input", metavar="FILE", help="Run detection from a JSON file")
    args = parser.parse_args()

    detector = HallucinationDetector()

    if args.demo:
        run_demo(detector)
    elif args.test:
        run_single_test(detector, args.test)
    elif args.input:
        run_from_file(detector, args.input)
    else:
        run_interactive(detector)


if __name__ == "__main__":
    main()
