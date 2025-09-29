# Frontend Environment Setup

This document explains how to configure the frontend to work with different backend environments.

## Environment Files

- **`.env.local`** - Points to local backend (http://localhost:8001)
- **`.env.production`** - Points to Railway backend (https://ml-copilot-production.up.railway.app)
- **`.env`** - Active environment file (copied from one of the above)

## Quick Setup

### Option 1: Use the PowerShell Script (Recommended)

```powershell
# Switch to development mode (local backend)
.\switch-env.ps1 dev

# Switch to production mode (Railway backend)
.\switch-env.ps1 prod
```

### Option 2: Manual Setup

```powershell
# For local development
Copy-Item ".env.local" ".env" -Force

# For production
Copy-Item ".env.production" ".env" -Force
```

## Running the Frontend

After setting up the environment:

```bash
npm run dev
```

## Environment Details

### Development Mode (Local Backend)
- **Backend URL**: http://localhost:8001
- **Use Case**: Testing with your local backend
- **Requirements**: Local backend must be running on port 8001

### Production Mode (Railway Backend)
- **Backend URL**: https://ml-copilot-production.up.railway.app
- **Use Case**: Testing with deployed Railway backend
- **Requirements**: Railway backend must be deployed and running

## Troubleshooting

1. **Frontend can't connect to backend**: Check if the backend is running on the correct port
2. **CORS errors**: Ensure the backend has proper CORS configuration
3. **Environment not switching**: Make sure you're copying the correct `.env` file

## Current Configuration

- **Development**: Points to local backend on port 8001
- **Production**: Points to Railway backend
- **Default**: Falls back to localhost:8000 if no environment is set
