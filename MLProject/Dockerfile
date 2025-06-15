# ================================
# Base Image
# ================================
FROM python:3.12-slim

# ================================
# Working Directory
# ================================
WORKDIR /app

# ================================
# Copy All Project Files
# ================================
COPY . .

# ================================
# Install Dependencies
# ================================
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        pandas \
        numpy \
        scikit-learn \
        mlflow \
        matplotlib \
        seaborn \
        joblib

# ================================
# Default Command
# ================================
CMD ["python", "modelling.py"]
