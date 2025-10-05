from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client,StdioServerParameters
import asyncio
from contextlib import AsyncExitStack
from typing import List,Dict,Any
import logging
import re

logger = logging.getLogger(__name__)

class MCPClient:
    def __init__(self):
        self.exit_stacks = {}
        self.sessions = {}

    async def connect_sse(self,uri:str,server_id:str|None=None):
        exit_stack = AsyncExitStack()
        server_id = server_id or uri
        streams_context = sse_client(url=uri)
        streams = await exit_stack.enter_async_context(streams_context)
        session = await exit_stack.enter_async_context(ClientSession(*streams))
        await session.initialize()
        self.exit_stacks[server_id] = exit_stack
        self.sessions[server_id] = session

    async def connect_stdio(self,command,args:List[str],server_id:str|None = None):
        server_params = StdioServerParameters(command=command,args=args)
        exit_stack = AsyncExitStack()
        server_id = server_id or command
        read,write = await exit_stack.enter_async_context(stdio_client(server_params))
        session = await exit_stack.enter_async_context(ClientSession(read,write))
        await session.initialize()
        self.exit_stacks[server_id] = exit_stack
        self.sessions[server_id] = session

    async def disconnect_server(self,server_id:str = ''):
        if server_id:
            if server_id in self.sessions:
                try:
                    exit_stack:AsyncExitStack|None = self.exit_stacks.get(server_id,None)
                    if exit_stack:
                        try:
                            await exit_stack.aclose()
                        except Exception as e:
                            logger.warning(f'{server_id}: {str(e)}')
                    self.sessions.pop(server_id,None)
                    self.exit_stacks.pop(server_id,None)

                except Exception as e:
                    logger.warning(f'{server_id}: {str(e)}')
        else:
            for server_id in list(self.sessions.keys()):
                await self.disconnect_server(server_id)
        logger.info(f'关闭了所有mcp服务')

class MCPTools(MCPClient):
    def __init__(self):
        super().__init__()
        self.tools = []
    async def list_tools(self):
        tools = []
        for ss_name,session in self.sessions.items():
            for tool in (await session.list_tools()).tools:
                tools.append({
                    'type':'function',
                    'function':{
                        'name':f'{ss_name}__{tool.name}',
                        'description':tool.description,
                        'parameters':tool.inputSchema,
                    }
                })
        return tools
    async def connect_stdio(self, command, args, server_id = None):
        response = await super().connect_stdio(command, args, server_id)
        self.tools = await self.list_tools()
        return response
    async def connect_sse(self, uri, server_id = None):
        response = await super().connect_sse(uri, server_id)
        self.tools = await self.list_tools()
        return response
    async def disconnect_server(self, server_id = ''):
        response = await super().disconnect_server(server_id)
        self.tools = await self.list_tools()
        return response

    async def call_tool(self,tool_name:str,tool_args:Dict[str,Any]):
        ss_name = re.match('^.+?__',tool_name).group()[:-2]
        tool_name = re.sub('^.+?__','',tool_name)
        return await self.sessions[ss_name].call_tool(tool_name,tool_args)
    
    def __contains__(self,tool_name):
        tools = self.tools
        for tool in tools:
            if tool_name == tool['function']['name']:
                return True
        return False

if __name__ == '__main__':
    async def main():
        mcp_client = MCPClient()
        await mcp_client.connect_sse('http://127.0.0.1:8000/mcp/sse','mcp')
        response = await mcp_client.sessions['mcp'].list_tools()
        print(response.tools)
    asyncio.run(main())

            