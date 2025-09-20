from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal

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

class Operation(BaseModel):
    op: str
    col: Optional[str] = None
    x: Optional[str] = None
    y: Optional[str] = None
    sep: Optional[str] = None
    as_: Optional[List[str]] = Field(default=None, alias="as")
    extras: Optional[List[str]] = None
    log: Optional[bool] = None

class Plan(BaseModel):
    intent: Literal["chart", "table"]
    title: str
    operations: List[Operation] = Field(default_factory=list)
    vega_lite: Optional[Dict[str, Any]] = None