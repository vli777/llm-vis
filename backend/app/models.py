from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional, Literal, Union


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
    """Backend data operation to execute before visualization."""
    op: Literal["value_counts", "explode_counts", "scatter_data", "corr_pair"]
    col: Optional[str] = Field(None, description="Column name for value_counts/explode_counts")
    x: Optional[str] = Field(None, description="X column for scatter_data/corr_pair")
    y: Optional[str] = Field(None, description="Y column for scatter_data/corr_pair")
    sep: Optional[str] = Field(None, description="Separator regex for explode_counts")
    as_: Optional[List[str]] = Field(None, alias="as", description="Output column names")
    extras: Optional[List[str]] = Field(None, description="Extra columns to include")
    log: Optional[bool] = Field(None, description="Apply log transform for scatter_data")
    limit: Optional[int] = Field(None, description="Limit results for value_counts/explode_counts")

    @field_validator('op')
    @classmethod
    def validate_op(cls, v: str) -> str:
        """Ensure op is one of the supported operations."""
        valid_ops = {"value_counts", "explode_counts", "scatter_data", "corr_pair"}
        if v not in valid_ops:
            raise ValueError(f"op must be one of {valid_ops}, got '{v}'")
        return v


class JSONPatchOperation(BaseModel):
    """RFC 6902 JSON Patch operation."""
    op: Literal["add", "remove", "replace", "move", "copy", "test"]
    path: str = Field(..., description="JSON Pointer path (e.g., '/encoding/x/field')")
    value: Optional[Any] = Field(None, description="Value for add/replace operations")
    from_: Optional[str] = Field(None, alias="from", description="Source path for move/copy")


class Plan(BaseModel):
    """Structured plan for creating or updating a visualization."""
    action: Literal["create", "update"] = Field(
        default="create",
        description="Whether to create new visualization or update existing"
    )
    type: Literal["chart", "table"] = Field(
        default="chart",
        alias="intent",
        description="Type of visualization to create"
    )
    title: str = Field(..., description="Concise title (max 10 words)")
    operations: List[Operation] = Field(
        default_factory=list,
        description="Backend operations to execute before visualization (use sparingly, prefer Vega-Lite transforms)"
    )
    vega_lite: Optional[Dict[str, Any]] = Field(
        None,
        description="Vega-Lite specification (required when type='chart' and action='create')"
    )
    patch: Optional[List[JSONPatchOperation]] = Field(
        None,
        description="JSON Patch operations for action='update'"
    )
    targetId: Optional[str] = Field(
        None,
        description="ID of visualization to update (for action='update')"
    )
    target: Optional[Literal["last"]] = Field(
        None,
        description="Target selector (e.g., 'last') for action='update'"
    )

    @field_validator('title')
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Ensure title is concise."""
        if len(v.split()) > 15:
            raise ValueError("Title should be concise (max 15 words)")
        return v

    def model_post_init(self, __context):
        """Validate cross-field constraints after model initialization."""
        # Validate vega_lite is provided when needed
        if self.action == 'create' and self.type == 'chart' and self.vega_lite is None:
            raise ValueError("vega_lite is required when action='create' and type='chart'")

    model_config = {
        "populate_by_name": True,  # Allow both 'type' and 'intent' as field names
    }
