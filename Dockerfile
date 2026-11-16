# Python 3.14 image satisfies requires-python (>=3.12); uv uses the system interpreter below
FROM python:3.14-slim

# Set working directory
WORKDIR /app

# Install system dependencies (build-essential provides cc for sdist builds like httptools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency definition files
COPY pyproject.toml uv.lock .python-version ./

# Use the image's system Python instead of downloading a newer one
ENV UV_PYTHON_PREFERENCE=system

# Install dependencies using uv
RUN uv sync --frozen --no-install-project

# Copy project source code, models, and application
COPY src/ ./src/
COPY models/ ./models/
COPY app/ ./app/

# Expose Streamlit default port
EXPOSE 8501

# Healthcheck for Streamlit
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Default command to run the Streamlit app
ENTRYPOINT ["uv", "run", "streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
