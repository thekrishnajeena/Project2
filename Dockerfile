FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# 1) Install system dependencies required by Chromium and Playwright
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg wget fonts-liberation \
    libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 \
    libxcomposite1 libxdamage1 libxrandr2 libxss1 libasound2 \
    libgbm1 libgtk-3-0 libpangocairo-1.0-0 libpango-1.0-0 \
 && rm -rf /var/lib/apt/lists/*

# 2) Copy & install Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 3) Install Playwright browsers (and optionally their OS deps)
#    --with-deps downloads browsers and also attempts to install needed OS packages.
#    If your base image lacks apt, you may use `playwright install chromium` instead.
RUN python -m playwright install --with-deps

# 4) Copy app files
COPY . /app

# 5) Ensure HF expected port
ENV PORT=7860

# 6) Run Uvicorn pointing to app.main:app (module path)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
