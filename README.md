# Agri-GNN - Setup and Execution Instructions

## Setup Instructions

1. Extract or clone the project folder from GitHub.

2. Install Docker Desktop from: https://www.docker.com/products/docker-desktop/

3. Open a terminal/command prompt and run:
   ```
   wsl --update
   ```


4. Restart your system.

5. After restart, open Docker Desktop and ensure it is running.

6. Open the project folder in VS Code.

7. Navigate to the `smart-farm-dash` directory:
   ```
   cd smart-farm-dash
   ```


## Execution Instructions

### First-Time Run

1. Open a terminal/command prompt in the `smart-farm-dash` directory.

2. Execute:
   ```
   docker compose up --build
   ```

3. Wait for all containers to start successfully. This may take several minutes on the first run as Docker builds the images and installs dependencies.

### Subsequent Runs

1. Open a terminal/command prompt in the `smart-farm-dash` directory.

2. Execute:
   ```
   docker compose up
   ```

3. Wait for all containers to start successfully.

### Accessing the Application

After containers start successfully, open your browser and visit:
```
http://localhost:3000
```

The application will be accessible at this address. The frontend service runs on port 3000, the backend API on port 5000, and the ML service on port 8000.

---
