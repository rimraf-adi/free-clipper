import sys
from datetime import datetime

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[35m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

def log_info(stage: str, message: str):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"{Colors.CYAN}[{ts}] [{stage}]{Colors.RESET} {message}")

def log_success(stage: str, message: str):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"{Colors.GREEN}[{ts}] [{stage}] ✔ {message}{Colors.RESET}")

def log_warning(stage: str, message: str):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"{Colors.YELLOW}[{ts}] [{stage}] ⚠️  {message}{Colors.RESET}")

def log_error(stage: str, message: str):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"{Colors.RED}[{ts}] [{stage}] ✖ {message}{Colors.RESET}")

def log_step(stage: str, message: str):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"{Colors.MAGENTA}[{ts}] [{stage}] ➔ {Colors.RESET}{message}")

def print_header(title: str):
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*64}")
    print(f"  🚀  {title}")
    print(f"{'='*64}{Colors.RESET}\n")

def print_stage_banner(stage_num: int, title: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}─── STAGE {stage_num}: {title} ───{Colors.RESET}\n")

def print_summary_box(title: str, items: list):
    print(f"\n{Colors.BOLD}{Colors.GREEN}{'='*64}")
    print(f"  ✨  {title}")
    print(f"{'='*64}{Colors.RESET}")
    for item in items:
        print(f"  {Colors.BOLD}{Colors.CYAN}•{Colors.RESET} {item}")
    print(f"{Colors.BOLD}{Colors.GREEN}{'='*64}{Colors.RESET}\n")
