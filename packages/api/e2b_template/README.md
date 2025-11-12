# E2B Template Setup for ML Libraries

This template provides a Python sandbox with PyTorch and ML libraries pre-installed.

## Prerequisites

1. E2B API Key (already configured in `.env`)
2. Node.js/npm (for E2B CLI)

## Setup Steps

### 1. Install E2B CLI

```bash
npm install -g @e2b/cli
```

### 2. Login to E2B

```bash
e2b auth login
```

Enter your E2B API key when prompted: `e2b_2187...` (from your `.env` file)

### 3. Build and Deploy the Template

```bash
cd packages/api/e2b_template
e2b template build
```

This will:
- Build the Docker image with PyTorch and ML libraries
- Upload it to E2B
- Return a **template ID** (looks like: `abc123xyz`)

### 4. Update the Code

After building, you'll get a template ID. Update `packages/api/agent/tools/code_executor_tool.py`:

```python
# Change this line:
with E2BClass.create() as sandbox:

# To this (with your template ID):
with E2BClass.create(template="YOUR_TEMPLATE_ID_HERE") as sandbox:
```

Or better yet, add it to your `.env` file:

```bash
E2B_TEMPLATE_ID=your_template_id_here
```

And update the code to read from environment:

```python
template_id = os.getenv("E2B_TEMPLATE_ID")
with E2BClass.create(template=template_id) as sandbox:
```

## What's Included

The template includes:
- **NumPy** 1.24.3 - Numerical computing
- **Pandas** 2.0.3 - Data analysis
- **Matplotlib** 3.7.2 - Plotting/visualization
- **Scikit-learn** 1.3.0 - Machine learning
- **SciPy** 1.11.2 - Scientific computing

## What's NOT Included

- **PyTorch** - Too large (670MB+) for E2B disk limits
- **TensorFlow** - Also too large for E2B
  
For testing PyTorch/TensorFlow code, the agent will need to explain conceptually rather than execute.

## Notes

- Build time: ~1 minute
- Template updates: Re-run `e2b template create ml-copilot-sandbox` when you change packages
- The agent can still test NumPy/Pandas/Scikit-learn code, just not deep learning frameworks

## Alternative: Quick Test

If you don't want to create a custom template yet, you can:
1. Keep the code execution tool as-is for basic Python
2. Focus on the documentation search capabilities
3. Build the ML template later when needed

