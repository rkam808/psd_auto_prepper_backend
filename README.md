# PSD Auto Prepper

This project is a web application for automatically preparing PSD files. It consists of a backend, a frontend, and a machine learning service.

## Running with Docker

The entire application can be run using Docker and Docker Compose.

### Prerequisites

*   Docker
*   Docker Compose

### Setup

1.  **Set the Rails Master Key:** The backend service requires the `RAILS_MASTER_KEY` environment variable. You can create a `.env` file in the project root with your key. A command is provided below to extract it from the development credentials and create the `.env` file.

    ```bash
    echo "RAILS_MASTER_KEY=$(cat backend/config/master.key)" > .env
    ```

2.  **Build and run the containers:**

    ```bash
    docker-compose up --build
    ```

    This command will build the Docker images for each service and start them.

### Accessing the application

Once the containers are running, you can access the frontend at [http://localhost:8080](http://localhost:8080).

The services are configured as follows:

*   **Frontend:** Runs on port `8080` and serves the React application.
*   **Backend:** The Rails API is available at `http://localhost:3000`. The frontend proxies requests to `/api` to this service.
*   **ML Service:** The Python API is not directly exposed. The frontend proxies requests to `/ml` to this service.
