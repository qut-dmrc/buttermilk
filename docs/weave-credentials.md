# Weave Credentials Configuration

This document describes how to configure Weave (Weights & Biases) credentials for deployment scenarios where interactive login is not possible.

## Overview

The Buttermilk framework now supports automatic loading of Weave credentials from environment variables or secret management systems. This eliminates the need for interactive login flows during deployment.

## Credential Loading Order

Credentials are loaded in the following priority order:

1. **Environment Variables** (highest priority)
2. **Secret Manager** (fallback)
3. **Graceful Degradation** (mock client if no credentials available)

## Environment Variables

Set the following environment variables for Weave authentication:

```bash
export WANDB_API_KEY="your-wandb-api-key"
export WANDB_PROJECT="your-project-name"
export WANDB_ENTITY="your-entity-name"  # optional
```

### Example: Docker Deployment

```dockerfile
ENV WANDB_API_KEY="your-api-key"
ENV WANDB_PROJECT="my-research-project"
ENV WANDB_ENTITY="my-organization"
```

### Example: Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: buttermilk-app
spec:
  template:
    spec:
      containers:
      - name: buttermilk
        image: buttermilk:latest
        env:
        - name: WANDB_API_KEY
          valueFrom:
            secretKeyRef:
              name: wandb-credentials
              key: api-key
        - name: WANDB_PROJECT
          value: "my-research-project"
        - name: WANDB_ENTITY
          value: "my-organization"
```

## Secret Manager Integration

If environment variables are not set, Buttermilk will attempt to load credentials from the configured secret manager (e.g., Google Cloud Secret Manager).

The system looks for these keys in the shared credentials secret:

- `WANDB_API_KEY`
- `WANDB_PROJECT`
- `WANDB_ENTITY`

### Example: Google Cloud Secret Manager

Store your credentials in the secret configured for shared credentials:

```json
{
  "WANDB_API_KEY": "your-wandb-api-key",
  "WANDB_PROJECT": "your-project-name", 
  "WANDB_ENTITY": "your-entity-name"
}
```

## Mixed Configuration

You can mix environment variables and secret manager:

- Set `WANDB_API_KEY` as an environment variable (for security)
- Store `WANDB_PROJECT` and `WANDB_ENTITY` in secret manager (for flexibility)

Environment variables always take precedence over secret manager values.

## Graceful Fallback

If no credentials are available:

- Weave initialization will be attempted but may fail
- A mock Weave client will be returned instead of crashing
- All weave operations will be no-ops, allowing the application to continue
- Warning messages will be logged

## Getting Your WANDB API Key

1. Sign up for a Weights & Biases account at https://wandb.ai
2. Go to your account settings
3. Find the API key section
4. Copy your API key

## Testing Your Configuration

To verify your credentials are working:

```python
# This should not require interactive login
import weave
client = weave.init("test-project")
print("Weave initialized successfully!")
```

## Troubleshooting

### Common Issues

1. **Interactive login prompt appears**
   - Check that `WANDB_API_KEY` is set correctly
   - Verify the API key is valid

2. **Permission denied errors**
   - Check that `WANDB_PROJECT` matches a project you have access to
   - Verify `WANDB_ENTITY` is correct (if specified)

3. **Mock client is being used**
   - Check logs for credential loading errors
   - Verify secret manager access if using secrets

### Debug Logging

Enable verbose logging to see credential loading details:

```yaml
# In your configuration
bm:
  logger_cfg:
    verbose: true
```

This will show:
- Whether credentials were loaded from environment or secrets
- Any errors during credential loading
- Weave initialization success/failure

## Security Considerations

- Never commit API keys to source control
- Use secure secret management for production deployments
- Rotate API keys regularly
- Use least-privilege access for Weave projects

## Backward Compatibility

This change is fully backward compatible:
- Existing interactive login flows continue to work
- No configuration changes required for development environments
- Mock client ensures applications don't crash without credentials