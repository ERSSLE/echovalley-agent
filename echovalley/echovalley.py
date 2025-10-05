from typing import Dict,List
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import JsonOutputParser,StrOutputParser

from config import agent_settings
from browser_tool import BrowserTool,BingSearch
from python_tool import PythonExecutor
from base_tool import BaseTool
from mcp_client import MCPTools
from prompts import (
    system_template,
    judge_template,
    plan_template,
    reasoning_template,
    exec_template,
    generate_template,
    result_template,
    get_date
)
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('echovalley.log',encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

def get_llm():
    llm = ChatOpenAI(base_url=agent_settings.base_url,
                     api_key=agent_settings.api_key,
                     model=agent_settings.model,
                     temperature=agent_settings.temperature)
    return llm

class EchoValley(BaseModel):
    
    judge_llm: Runnable|None = Field(default=None,description='判别器llm')
    plan_llm: Runnable|None = Field(default=None,description='计划器llm')
    reasoning_llm: Runnable|None = Field(default=None,description='推理器llm')
    exec_llm: Runnable|None = Field(default=None,description='执行器llm')
    generate_llm: Runnable|None = Field(default=None,description='生成器llm')
    summary_llm: Runnable|None = Field(default=None,description='汇总历史生成答案llm')
    tools: Dict[str,BaseTool] = Field(default_factory=lambda: {})
    mcptools: MCPTools|None = Field(default=None,description='mcp工具')

    class Config:
        arbitrary_types_allowed=True
        extra = 'allow'

    @classmethod
    async def create(cls,mcp:MCPTools|None=None):
        llm = get_llm()
        tools = [
            await BrowserTool.create(),
            # await PythonExecutor.create(),
            await BingSearch.create(host='127.0.0.1',port=8000), # 与web-fetch-mcp对应
        ]
        tools_schema = [tool.get_json_schema() for tool in tools] + mcp.tools
        tools = {tool.name:tool for tool in tools}
        llm_with_tools = llm.bind(tools=tools_schema,tool_choice='auto')

        judge_prompt = ChatPromptTemplate.from_messages([
            ('system',system_template),
            ('user',judge_template)
        ])
        judge_llm = judge_prompt | llm | JsonOutputParser()

        plan_prompt = ChatPromptTemplate.from_messages([
            ('system',system_template),
            ('user',plan_template)
        ])
        plan_llm = plan_prompt | llm_with_tools | StrOutputParser()

        reasoning_prompt = ChatPromptTemplate.from_messages([
            # ('system',system_template),
            MessagesPlaceholder(variable_name='history'),
            ('user',reasoning_template)
        ])
        reasoning_llm = reasoning_prompt | llm_with_tools | JsonOutputParser()

        exec_prompt = ChatPromptTemplate.from_messages([
            # ('system',system_template),
            MessagesPlaceholder(variable_name='history'),
            ('user',exec_template)
        ])
        exec_llm = exec_prompt | llm_with_tools

        generate_prompt = ChatPromptTemplate.from_messages([
            ('system',system_template),
            ('user',generate_template)
        ])
        generate_llm = generate_prompt | llm | StrOutputParser()

        summary_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name='history'),
            ('user',result_template)
        ])
        summary_llm = summary_prompt | llm | StrOutputParser()

        return cls(judge_llm=judge_llm,
                   plan_llm=plan_llm,
                   reasoning_llm=reasoning_llm,
                   exec_llm=exec_llm,
                   generate_llm=generate_llm,
                   summary_llm=summary_llm,
                   tools = tools,
                   mcptools=mcp,
                   )
    
    async def arun(self,query:str):
        judge_response = await self.judge_llm.ainvoke({'datestr':get_date(),'query':query})
        # logger.info(f'judge-response: {judge_response}')
        if '是' in judge_response['answer']:
            logger.info('该任务需要多步完成')
            plan_response = await self.plan_llm.ainvoke({'datestr':get_date(),'query':query})
            logger.info(f'完成该任务的计划: {plan_response}')
            history = [
                ('system',system_template.format(datestr=get_date())),
                ('user',f'原问题：{query}\n计划:{plan_response}')
            ]
            browser_context = None
            iter_num = 0
            while (iter_num < agent_settings.max_steps):
                reasoning_response = await self.reasoning_llm.ainvoke({'history':history})
                logger.info(f'iter_num:{iter_num}, reasoning-response:{reasoning_response}')
                if ('退出' not in reasoning_response['next_prompt']) or (len(reasoning_response['next_prompt']) > 4):
                    history.append(('user',reasoning_response['current_situation']))
                    # reasoning_response['history'] = history
                    exec_response = await self.exec_llm.ainvoke({'history':history,
                                                                 'next_prompt':reasoning_response['next_prompt']})
                    history.append(('user',f'{reasoning_response['next_prompt']}'))
                    # history = (await self.exec_llm.steps[0].ainvoke(reasoning_response)).messages
                    # logger.info(f'iter_num:{iter_num}, exec-response-content:{exec_response.content}, exec-response-tool_calls:{exec_response.tool_calls}')
                    history.append(exec_response)
                    tool_responses,have_browser,browser_context = await self.call_tools(exec_response.tool_calls,browser_context)
                    # logger.info(f'iter_num:{iter_num}, have_browser:{have_browser}, tool_responses:{tool_responses}')
                    for tool_response in tool_responses:
                        history.append(ToolMessage(**tool_response))
                    if have_browser:
                        observation = await self.get_browser_observation(browser_context)
                        # logger.info(f'iter_num:{iter_num}, observation:{observation}')
                        history.append(('user',f'观察：{observation}'))
                else:
                    break
                iter_num += 1
                # self.print_history(idx=iter_num,history=history)
            if browser_context is not None:
                self.tools['browser']._release_context_by_id(id=browser_context['id'])
            generate_response = await self.summary_llm.ainvoke({'history':history})
        else:
            generate_response = await self.generate_llm.ainvoke({'datestr':get_date(),'query':query})
        with open('output.md','w',encoding='utf8') as file:
            file.write(generate_response)
        return generate_response
    
    async def cleanup(self):
        for name,tool in self.tools.items():
            await tool.cleanup()
        await self.mcptools.disconnect_server()

    async def get_browser_observation(self,browser_context):
        return await self.tools['browser']._get_current_browser_states(browser_context)
    
    async def call_tools(self,tool_calls,browser_context):
        tool_responses = []
        have_browser = False
        for tool_call in tool_calls:
            name = tool_call['name']
            args = tool_call['args']
            id = tool_call['id']
            print(name,args)
            if name in self.tools:
                if name == 'browser':
                    if browser_context is None:
                        browser_context = await self.tools[name]._get_idle_context()
                    args['context_id'] = browser_context['id']
                    have_browser = True
                tool_output = await self.tools[name](**args)
                tool_output_str = f'tool_message: {tool_output.message['result']}, state={tool_output.state}'
            elif self.mcptools and name in self.mcptools:
                mcp_output = await self.mcptools.call_tool(name,args)
                mcp_output = '\n\n'.join([content.text for content in mcp_output.content])
                tool_output_str = f'tool_message: {mcp_output}'
            tool_responses.append({'content':tool_output_str,'tool_call_id':id})
        return tool_responses,have_browser,browser_context
    
    def print_history(self,idx,history):
        print('=='*20 + f'{idx}' + '=='*20)
        for record in history:
            print(record)
            print()

if __name__ == '__main__':
    import asyncio
    async def main():
        mcp = MCPTools()
        # await mcp.connect_sse('http://127.0.0.1:8000/mcp/sse','mcp') # 这里可以连接其它服务
        echovalley = await EchoValley.create(mcp=mcp)
        while True:
            query = input('输入你的任务(如果退出使用exit)：')
            if query.strip() == 'exit':
                break
            else:
                res = await echovalley.arun(query)
                print('最终结果：',res)
        await echovalley.cleanup()
    asyncio.run(main())
