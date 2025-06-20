# ================================
# Base Image
# ================================
FROM python:3.12-slim

# ================================
# Working Directory
# ================================
WORKDIR /app

# ================================
# Arguments for credentials (optional)
# ================================
ARG DAGSHUB_USERNAME
ARG DAGSHUB_TOKEN

# Set ENV for runtime use
ENV MLFLOW_TRACKING_URI="file:///app/mlruns"
ENV DAGSHUB_USERNAME=${DAGSHUB_USERNAME}
ENV DAGSHUB_TOKEN=${DAGSHUB_TOKEN}

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
        joblib \
        dagshub

# ================================
# Copy Project Files
# ================================
COPY MLProject/ .

# Ensure folders exist
RUN mkdir -p mlruns output

# ================================
# Entrypoint and CMD
# ================================
ENTRYPOINT ["python", "-u"]
CMD ["Modelling.py"]