"""
프롬프트 템플릿 경로 상수 모음.

CWD가 아니라 이 파일 위치(__file__) 기준으로 절대경로를 계산하므로, 패키지로
쪼개거나 다른 디렉터리에서 실행해도 경로가 깨지지 않는다. 빌더/노드들은
하드코딩 대신 여기 상수만 import 한다.
"""
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent / "Prompts"


def _p(name: str) -> str:
    """Prompts 디렉터리 기준 파일명을 절대경로 문자열로 변환한다."""
    return str(PROMPTS_DIR / name)


# --- FGI (Realtime) ---
MULTI_FGI_SYSTEM = _p("multi_fgi_system.md")
MULTI_FGI_USER = _p("multi_fgi_user.md")
FREE_TALK_SYSTEM = _p("free_talk_system.md")
FREE_TALK_USER = _p("free_talk_user.md")

# --- Summary ---
DIALOGUE_SUMMARY_SYSTEM = _p("dialogue_summary_system.md")
DIALOGUE_SUMMARY_USER = _p("dialogue_summary_user.md")
ML_SUMMARY_SYSTEM = _p("ml_summary_system.md")
ML_SUMMARY_USER = _p("ml_summary_user.md")
PIXEL_SUMMARY_SYSTEM = _p("pixel_summary_system.md")
PIXEL_SUMMARY_USER = _p("pixel_summary_user.md")

# --- Mass Simulation ---
SIMULATION_SYSTEM = _p("simulation_system.md")
SIMULATION_USER = _p("simulation_user.md")
CAMPAIGN_EXTRACTION = _p("campaign_extraction.md")
JUDGE_SYSTEM = _p("judge_system.md")
JUDGE_USER = _p("judge_user.md")
