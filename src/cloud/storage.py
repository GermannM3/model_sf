from typing import Optional, List, BinaryIO
import boto3
import google.cloud.storage
from azure.storage.blob import BlobServiceClient
from pathlib import Path

class CloudStorage:
    def __init__(self, provider: str = "aws"):
        self.provider = provider
        self.client = self._initialize_client()
        
    def _initialize_client(self):
        if self.provider == "aws":
            return boto3.client('s3')
        elif self.provider == "gcp":
            return google.cloud.storage.Client()
        elif self.provider == "azure":
            connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
            return BlobServiceClient.from_connection_string(connection_string)
        else:
            raise ValueError(f"Unsupported cloud provider: {self.provider}")
            
    def upload_file(self, file_path: str, bucket: str, remote_path: Optional[str] = None):
        if self.provider == "aws":
            remote_path = remote_path or Path(file_path).name
            self.client.upload_file(file_path, bucket, remote_path)
        elif self.provider == "gcp":
            bucket = self.client.bucket(bucket)
            blob = bucket.blob(remote_path or Path(file_path).name)
            blob.upload_from_filename(file_path)
        elif self.provider == "azure":
            container_client = self.client.get_container_client(bucket)
            with open(file_path, "rb") as data:
                container_client.upload_blob(
                    name=remote_path or Path(file_path).name,
                    data=data
                )
                
    def download_file(self, bucket: str, remote_path: str, local_path: str):
        if self.provider == "aws":
            self.client.download_file(bucket, remote_path, local_path)
        elif self.provider == "gcp":
            bucket = self.client.bucket(bucket)
            blob = bucket.blob(remote_path)
            blob.download_to_filename(local_path)
        elif self.provider == "azure":
            container_client = self.client.get_container_client(bucket)
            with open(local_path, "wb") as file:
                data = container_client.download_blob(remote_path).readall()
                file.write(data)
                
    def list_files(self, bucket: str, prefix: Optional[str] = None) -> List[str]:
        if self.provider == "aws":
            response = self.client.list_objects_v2(Bucket=bucket, Prefix=prefix or "")
            return [obj['Key'] for obj in response.get('Contents', [])]
        elif self.provider == "gcp":
            bucket = self.client.bucket(bucket)
            return [blob.name for blob in bucket.list_blobs(prefix=prefix)]
        elif self.provider == "azure":
            container_client = self.client.get_container_client(bucket)
            return [blob.name for blob in container_client.list_blobs(name_starts_with=prefix)] 