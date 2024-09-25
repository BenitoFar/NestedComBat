# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app
# RUN chmod -R 777 /app

# Install any Python dependencies required by your script
# Ensure you have a requirements.txt file listing the dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Run the Python script when the container launches
CMD ["python", "./src/run.py"]