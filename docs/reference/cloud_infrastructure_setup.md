# IAM permissions

```
# grant devops role (almost) all of storage.objectUser's permissions:
gcloud iam roles update devops \
    --project=${GOOGLE_CLOUD_PROJECT} \
    --add-permissions="storage.folders.create,storage.folders.delete,storage.folders.get,storage.folders.list,storage.folders.rename,storage.managedFolders.create,storage.managedFolders.delete,storage.managedFolders.get,storage.managedFolders.list,storage.multipartUploads.abort,storage.multipartUploads.create,storage.multipartUploads.list,storage.multipartUploads.listParts,storage.objects.create,storage.objects.delete,storage.objects.get,storage.objects.list,storage.objects.move,storage.objects.restore,storage.objects.update" 
```

## Copying the permissions from a predefined role to a custom role:
PERMISSIONS=$(gcloud iam roles describe "$SOURCE_ROLE" --format="value(includedPermissions)" | tr ';' ',')
gcloud iam roles update $ROLE --project=$GOOGLE_CLOUD_PROJECT --add-permissions=$PERMISSIONS

## Workload identity pools

Create pool:
```shell
gcloud iam workload-identity-pools create --location=global --project=prosocial-443205 --display-name=devops devops
```

Create provider for Github :
```
# Create a provider for Github 
gcloud iam workload-identity-pools providers create-oidc "github-actions" \
--project="prosocial-443205" --location="global" --workload-identity-pool="devops" --display-name="github actions provider" \
--attribute-mapping="google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.aud=assertion.aud,attribute.repository_owner=assertion.repository_owner,attribute.repository=assertion.repository" \
   --attribute-condition="attribute.repository_owner==assertion.repository_owner&&attribute.repository==assertion.repository" \
   --issuer-uri="https://token.actions.githubusercontent.com"
```

