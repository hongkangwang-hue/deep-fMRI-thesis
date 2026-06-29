"""
Mm6  figures — placeholder
见 milestone/ 计划文档
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config_loader import load_config

if __name__ == "__main__":
    cfg = load_config()
    print("Not implemented yet")
