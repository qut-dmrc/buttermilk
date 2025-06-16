# Buttermilk Quick Start Guide

## ✅ Ready to Use - Two Working Examples

### 1. Trans Flow (Hierarchical Composition)
Uses Hydra's composition to pull in reusable components:

```bash
# Console mode
uv run python -m buttermilk.runner.cli run=console +flow=trans_clean

# With a prompt
uv run python -m buttermilk.runner.cli run=console +flow=trans_clean +prompt="Analyze this article"
```

### 2. Tox Flow (All-in-One)
Everything defined in a single file:

```bash
# Console mode  
uv run python -m buttermilk.runner.cli run=console +flow=tox_allinone

# With a prompt
uv run python -m buttermilk.runner.cli run=console +flow=tox_allinone +prompt="Check this content for toxicity"
```

## 🚀 Start the API Server

```bash
# Start API server (runs on http://localhost:8000)
uv run python -m buttermilk.runner.cli run=api

# Then use the web interface or make HTTP requests
curl -X POST http://localhost:8000/flow/trans_clean \
  -H "Content-Type: application/json" \
  -d '{"flow": "trans_clean", "prompt": "Your text here"}'
```

## 📚 Full Documentation

See `/workspaces/buttermilk/conf/README_FLOWS.md` for complete documentation including:
- All command line options
- API endpoints
- Configuration hierarchy
- Troubleshooting

## 🔧 Configuration Files

- **`conf/flows/trans_clean.yaml`** - Hierarchical composition example
- **`conf/flows/tox_allinone.yaml`** - All-in-one configuration example  
- **`conf/README_FLOWS.md`** - Complete documentation

The configuration system now supports both approaches so you can choose what works best for your needs!