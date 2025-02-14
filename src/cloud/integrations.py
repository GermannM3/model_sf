from typing import Optional, Dict, Any
import requests
import json
from dataclasses import dataclass

@dataclass
class CloudServiceConfig:
    service_name: str
    api_key: str
    region: Optional[str] = None
    project_id: Optional[str] = None
    endpoint: Optional[str] = None

class CloudServiceIntegration:
    def __init__(self, config: CloudServiceConfig):
        self.config = config
        self.session = self._create_session()
        
    def _create_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update({
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        })
        return session
        
    async def call_cloud_function(self, function_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Вызывает облачную функцию"""
        if self.config.service_name == "aws_lambda":
            return await self._call_aws_lambda(function_name, payload)
        elif self.config.service_name == "google_cloud_functions":
            return await self._call_google_function(function_name, payload)
        elif self.config.service_name == "azure_functions":
            return await self._call_azure_function(function_name, payload)
        else:
            raise ValueError(f"Unsupported cloud service: {self.config.service_name}")
            
    async def _call_aws_lambda(self, function_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"https://lambda.{self.config.region}.amazonaws.com/2015-03-31/functions/{function_name}/invocations"
        response = self.session.post(url, json=payload)
        return response.json()
        
    async def _call_google_function(self, function_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"https://{self.config.region}-{self.config.project_id}.cloudfunctions.net/{function_name}"
        response = self.session.post(url, json=payload)
        return response.json()
        
    async def _call_azure_function(self, function_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.config.endpoint}/api/{function_name}"
        response = self.session.post(url, json=payload)
        return response.json() 