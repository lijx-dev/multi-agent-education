"""应用配置管理，通过环境变量加载。"""
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # 通义千问API
    dashscope_api_key: str = ""
    dashscope_model: str = "qwen-turbo"

    # 保留原配置（后续扩展用）
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    minimax_api_key: str = ""
    minimax_model: str = "MiniMax-M2.7"

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/edu_agent"
    redis_url: str = "redis://localhost:6379/0"

    # Server（新增：支持.env文件中的所有端口配置）
    api_port: int = 8000
    python_api_port: int = 8000
    java_api_port: int = 8080
    go_api_port: int = 8081
    frontend_port: int = 3000
    log_level: str = "INFO"

    # BKT算法参数
    bkt_p_init: float = 0.1
    bkt_p_transit: float = 0.15
    bkt_p_guess: float = 0.2
    bkt_p_slip: float = 0.1

    # 课程规划参数
    mastery_threshold: float = 0.6
    weakness_threshold: float = 0.3
    weakness_min_attempts: int = 3

    # 提示策略参数
    max_hint_level_1: int = 1
    max_hint_level_2: int = 3
    low_mastery_threshold: float = 0.15

    # 互动监测参数
    max_idle_seconds: int = 300
    max_consecutive_errors: int = 3
    max_session_minutes: int = 45
    boredom_accuracy_threshold: float = 0.9
    boredom_min_streak: int = 5
    encouragement_interval: int = 3

    # 关键修改：允许额外的环境变量，避免future errors
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"  # 忽略未定义的环境变量
    )

settings = Settings()