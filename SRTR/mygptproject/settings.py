from pathlib import Path
import os

# --- 基本路径 ---
BASE_DIR = Path(__file__).resolve().parent.parent

# --- 媒体 ---
MEDIA_ROOT = os.environ.get("MEDIA_ROOT", os.path.join(BASE_DIR.parent, "media"))
MEDIA_URL = "/media/"

# --- 静态资源 ---
STATIC_URL = "/static/"
STATICFILES_DIRS = [ BASE_DIR / "csv_files" ]  # 若没有 csv_files 目录可去掉
STATIC_ROOT = BASE_DIR / "staticfiles"         # 生产环境 collectstatic 目标

# --- 本地资料库 & 执行产物（新加，供我们的多Agent用） ---
DATA_REPO_ROOT = BASE_DIR / "data_repo"   # 本地"网站内容库"：dictionaries/concepts/simulation/docs
STORAGE_ROOT   = BASE_DIR / "storage"     # 执行产物输出：/storage/jobs/<id>/...
os.makedirs(DATA_REPO_ROOT, exist_ok=True)
os.makedirs(STORAGE_ROOT, exist_ok=True)

# --- 安全/调试 ---
SECRET_KEY = os.environ.get(
    "DJANGO_SECRET_KEY",
    "django-insecure-xi7_exxohnyd359-4fifq96qe@hs0)+u&v-h@aiio-tv94e25-",
)
DEBUG = os.environ.get("DJANGO_DEBUG", "True").lower() in ("true", "1", "yes")
ALLOWED_HOSTS = os.environ.get("DJANGO_ALLOWED_HOSTS", "*").split(",")

# --- Apps ---
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "gptapp",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "mygptproject.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [ BASE_DIR / "gptapp" / "templates" ],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "mygptproject.wsgi.application"

# --- DB ---
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

# --- 密码策略 ---
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

# --- 国际化 ---
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

# --- 日志 ---
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {"class": "logging.StreamHandler"},
    },
    "root": {"handlers": ["console"], "level": "DEBUG"},
}

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# 上传大小限制
DATA_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024  # 10MB
