# Config Management for Buttermilk

- **Hydra** for composition & user flexibility (researchers mix/match components)
- **Pydantic** for validation at component boundaries
- **Cloud secrets** for shared API keys and LLM configs
- **Environment variables + local overrides** for sensitive data


## Hydra Composition & Overrides

**Config groups structure:**

```
conf/
├── config.yaml           # Main config with defaults
├── data/
│   ├── postgres.yaml
│   └── s3.yaml
└── analysis/
    ├── sentiment.yaml
    └── topic.yaml
```

**Composition magic:**

- Hydra merges all configs in `defaults` list
- Later entries override earlier ones
- CLI overrides beat everything

### Config Pattern

**Public config (checked in):**

```yaml
# examples/config.yaml
data:
  _target_: mylib.BigQuerySource
  project: ${oc.env:GCP_PROJECT,YOUR_PROJECT_ID}
  dataset: research_data

storage:
  bucket: ${oc.env:GCS_BUCKET,your-bucket-name}
```

**Local overrides (gitignored):**

```yaml
# examples/config.local.yaml
data:
  project: my-real-project-123
storage:
  bucket: my-actual-bucket
```

**Hydra defaults:**

```yaml
defaults:
  - config
  - optional config.local
```

## Component Design

```python
from pydantic import BaseModel

class DataSourceConfig(BaseModel):
    project: str
    dataset: str

class DataSource:
    def __init__(self, **kwargs):
        self.config = DataSourceConfig(**kwargs)  # Pydantic validates
```
