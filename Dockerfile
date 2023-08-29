# Use the official Python base image
FROM python:3.11

# Set working directory
WORKDIR /app

# Copy source code into the container
COPY . /app

# Setup the environment variable for the connection string
ARG OPENAI_API_KEY
ENV OPENAI_API_KEY=$OPENAI_API_KEY

# Install requirements
RUN pip install -r requirements.txt

EXPOSE 8501

CMD streamlit run --server.port 8501 --logger.level=debug app.py