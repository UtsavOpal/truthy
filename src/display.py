"""
Terminal display utilities for the Hallucination Detection Framework.
Uses ANSI color codes for a rich CLI experience.
"""

from src.models import DetectionInput, HallucinationResult, HallucinationType

# ── ANSI color codes ──────────────────────────────────────────────────────────
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"
    GRAY    = "\033[90m"

    # Background
    BG_RED   = "\033[41m"
    BG_GREEN = "\033[42m"


# Type-specific colors
TYPE_COLORS = {
    "1A": C.RED,
    "1B": C.YELLOW,
    "2A": C.MAGENTA,
    "3A": C.BLUE,
}


class ResultDisplay:

    def print_banner(self):
        print(f"\n{C.CYAN}{C.BOLD}")
        print("  ╔══════════════════════════════════════════════════════════════╗")
        print("  ║         HALLUCINATION DETECTION FRAMEWORK  v1.0             ║")
        print("  ║                   Powered by OpenAI GPT-4o                  ║")
        print("  ╚══════════════════════════════════════════════════════════════╝")
        print(f"{C.RESET}")

        # Legend
        print(f"  {C.BOLD}Taxonomy:{C.RESET}")
        for ht in HallucinationType:
            color = TYPE_COLORS.get(ht.value, C.WHITE)
            print(f"  {color}[{ht.value}]{C.RESET} {ht.icon}  "
                  f"{C.BOLD}{ht.display_name}{C.RESET}"
                  f"{C.GRAY} – {ht.description}{C.RESET}")
        print()

    def show_result(self, inp: DetectionInput, result: HallucinationResult):
        """Print a full detection result to the terminal."""
        self._print_input_summary(inp)
        self._print_verdict(result)
        if result.is_hallucinated:
            self._print_types(result)
            self._print_hallucinated_elements(result)
        self._print_explanation(result)
        self._print_correct_answer(result)
        self._print_confidence_bar(result)

    # ── Private ───────────────────────────────────────────────────────────────

    def _print_input_summary(self, inp: DetectionInput):
        print(f"\n{C.GRAY}  {'─' * 63}{C.RESET}")
        print(f"  {C.BOLD}{C.WHITE}INPUT SUMMARY{C.RESET}")
        print(f"{C.GRAY}  {'─' * 63}{C.RESET}")
        if inp.paragraph:
            excerpt = inp.paragraph.strip()
            excerpt = excerpt[:200] + ("…" if len(excerpt) > 200 else "")
            print(f"  {C.GRAY}Paragraph :{C.RESET} {excerpt}")
        else:
            print(f"  {C.GRAY}Paragraph :{C.RESET} {C.DIM}(none – world knowledge used){C.RESET}")
        print(f"  {C.GRAY}Question  :{C.RESET} {C.CYAN}{inp.question}{C.RESET}")
        print(f"  {C.GRAY}Answer    :{C.RESET} {C.YELLOW}{inp.answer}{C.RESET}")

    def _print_verdict(self, result: HallucinationResult):
        print(f"\n{C.GRAY}  {'─' * 63}{C.RESET}")
        if result.is_hallucinated:
            print(f"  {C.BG_RED}{C.BOLD}{C.WHITE}  🚨  VERDICT: HALLUCINATION DETECTED  {C.RESET}")
        else:
            print(f"  {C.BG_GREEN}{C.BOLD}{C.WHITE}  ✅  VERDICT: OUTPUT IS CLEAN (NO HALLUCINATION)  {C.RESET}")
        print(f"{C.GRAY}  {'─' * 63}{C.RESET}")

    def _print_types(self, result: HallucinationResult):
        if not result.hallucination_types:
            return
        print(f"\n  {C.BOLD}Hallucination Type(s) Detected:{C.RESET}")
        for ht in result.hallucination_types:
            color = TYPE_COLORS.get(ht.value, C.WHITE)
            print(f"    {color}{C.BOLD}[{ht.value}]{C.RESET}  "
                  f"{ht.icon}  {C.BOLD}{ht.display_name}{C.RESET}")
            print(f"         {C.GRAY}{ht.description}{C.RESET}")

    def _print_hallucinated_elements(self, result: HallucinationResult):
        if not result.hallucinated_elements:
            return
        print(f"\n  {C.BOLD}Hallucinated Element(s):{C.RESET}")
        for elem in result.hallucinated_elements:
            print(f"    {C.RED}✗{C.RESET}  {elem}")

    def _print_explanation(self, result: HallucinationResult):
        if not result.explanation:
            return
        print(f"\n  {C.BOLD}Explanation:{C.RESET}")
        # Word-wrap at ~60 chars
        words = result.explanation.split()
        line = "    "
        for word in words:
            if len(line) + len(word) + 1 > 66:
                print(line)
                line = "    " + word + " "
            else:
                line += word + " "
        if line.strip():
            print(line)

    def _print_correct_answer(self, result: HallucinationResult):
        if not result.correct_answer:
            return
        print(f"\n  {C.BOLD}Correct Answer:{C.RESET}")
        print(f"    {C.GREEN}{result.correct_answer}{C.RESET}")

    def _print_confidence_bar(self, result: HallucinationResult):
        pct = result.confidence
        filled = round(pct / 5)           # out of 20 blocks
        bar = "█" * filled + "░" * (20 - filled)
        color = C.GREEN if pct >= 75 else C.YELLOW if pct >= 50 else C.RED
        print(f"\n  {C.BOLD}Confidence:{C.RESET}  "
              f"{color}[{bar}]{C.RESET}  {C.BOLD}{pct}%{C.RESET}")
        print(f"{C.GRAY}  {'─' * 63}{C.RESET}\n")
