FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose Gradio port
EXPOSE 7860

# Run the app
CMD ["python", "gradio_app.py"]