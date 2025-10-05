
from enum import Enum
from pydantic import BaseModel,Field
from typing import Any,Dict,Optional
from abc import ABC,abstractmethod
import asyncio
from utils import get_json_schema
import json

class ToolState(str,Enum):
    IDLE = 'IDLE'
    USING = 'USING'

class ToolError(Exception):
    def __init__(self,message):
        self.message = message

class ToolResult(BaseModel):
    message: Dict[str, Any] = Field(default_factory=dict,description='工具调用的结果')
    state: str = Field(default='成功',description='状态：成功 或 异常') # 成功 or 异常

class BaseTool(ABC,BaseModel):
    # 工具调用时使用的上下文信息，用于在多次调用之间传递工具状态
    name: str = ''
    base_context: Optional[Dict[str,Any]] = None
    lock: asyncio.Lock = asyncio.Lock()

    class Config:
        arbitrary_types_allowed = True
        # extra = 'allow'

    async def __call__(self, *args, **kwargs) -> Any:
        """按照给定的参数进行工具调用"""
        return await self.invoke(*args,**kwargs)

    @abstractmethod
    async def invoke(self,*args,**kwargs) -> Any:
        """执行工具调用的方法，需要具体实现。必须要包含方法文档与参数提示，这对生成正确的工具schema很重要
        """

    def get_json_schema(self):
        schema = get_json_schema(self.invoke)
        return json.loads(json.dumps(schema).replace('invoke',self.name))
    
    @abstractmethod
    async def create(cls, *args, **kwargs) -> 'BaseTool':
        """创建工具实例"""

    async def cleanup(self):
        """清除必要上下文"""
