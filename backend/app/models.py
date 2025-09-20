from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class NLQRequest(BaseModel):
    prompt: str
    clientContext: Optional[Dict[str, Any]] = None

class TableInfo(BaseModel):
    name: str
    rows: int

class TablePayload(BaseModel):
    columns: List[str]
    rows: List[Dict[str, Any]]

class NLQResponse(BaseModel):
    type: str  # 'chart' or 'table'
    title: Optional[str] = None
    spec: Optional[Dict[str, Any]] = None
    table: Optional[TablePayload] = None
