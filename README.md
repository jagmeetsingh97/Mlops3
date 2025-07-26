Purpose: Containerization + CI/CD automation.

Action:

Adds a Dockerfile to containerize the app.

Adds GitHub Actions workflow (ci.yml) to automate:

Model training

Docker build

Docker run for verification

DockerHub image push

Key Files: Dockerfile, predict.py, .github/workflows/ci.yml