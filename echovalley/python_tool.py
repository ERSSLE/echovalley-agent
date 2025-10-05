from base_tool import BaseTool
import sys
from io import StringIO
from typing import Dict
import multiprocessing
from base_tool import ToolResult

class PythonExecutor(BaseTool):
    name: str = 'python_executor'

    def _run_code(self, code: str, result_dict: dict, safe_globals: dict) -> None:
        original_stdout = sys.stdout
        try:
            output_buffer = StringIO()
            sys.stdout = output_buffer
            exec(code, safe_globals, safe_globals)
            result_dict["observation"] = output_buffer.getvalue()
            result_dict["success"] = True
        except Exception as e:
            result_dict["observation"] = str(e)
            result_dict["success"] = False
        finally:
            sys.stdout = original_stdout

    async def invoke(
        self,
        code: str,
        timeout: int = 5,
    ) -> Dict:
        """
        执行提供的Python代码并设置超时。
        注意：只有打印输出可见，函数返回值不会被捕获。使用打印语句查看结果。

        Args:
            code: 要执行的python代码
            timeout: 设置的执行超时(秒)

        Returns:
            Dict: 包含带有执行输出或错误消息的“output”以及“success”状态。
        """
        try:
            with multiprocessing.Manager() as manager:
                result = manager.dict({"observation": "", "success": False})
                if isinstance(__builtins__, dict):
                    safe_globals = {"__builtins__": __builtins__}
                else:
                    safe_globals = {"__builtins__": __builtins__.__dict__.copy()}
                proc = multiprocessing.Process(
                    target=self._run_code, args=(code, result, safe_globals)
                )
                proc.start()
                proc.join(timeout)

                # timeout process
                if proc.is_alive():
                    proc.terminate()
                    proc.join(1)
                    return ToolResult(message={'result':f'经过{timeout}秒后超时异常'},state='异常')
                result = dict(result)
                return ToolResult(message={'result':result['observation']},state='成功')
        except Exception as e:
            return ToolResult(message={'result':f'python interpreter 出现异常{str(e)}'},state='异常')

    @classmethod
    async def create(cls):
        return cls()
    
if __name__ == '__main__':
    import asyncio
    async def main():
        python = await PythonExecutor.create()
        a = await python('a=1\nprint(a)')
        print(a)
        print(python.get_json_schema())
    asyncio.run(main())