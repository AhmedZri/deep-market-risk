FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt requirements-package.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
RUN pip install --no-cache-dir --no-deps . && useradd --create-home --uid 10001 appuser && mkdir -p /app/artifacts /app/data && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000 8501
CMD ["python", "-m", "uvicorn", "market_risk.api:app", "--host", "0.0.0.0", "--port", "8000"]
