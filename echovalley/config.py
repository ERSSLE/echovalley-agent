from pydantic import BaseModel,Field
import tomllib
from pathlib import Path

def get_config_path():
    return Path(__file__).parent

ROOT = get_config_path()

class AgentSettings(BaseModel):
    base_url: str = Field(...,description='大预言模型的API base url')
    api_key: str = Field(...,description='API Key')
    model: str = Field(...,description='模型名称')
    temperature: float = Field(...,description='模型temperature')
    max_steps: int = 20
    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_toml_file(cls,file_path):
        with file_path.open('rb') as file_ob:
            config = tomllib.load(file_ob)
        return cls(base_url=config['LLM']['base_url'],
                   api_key=config['LLM']['api_key'],
                   model=config['LLM']['model'],
                   temperature=config['LLM']['temperature'],
                   max_steps=config['OTHER']['max_steps'],
                   )

agent_settings = AgentSettings.from_toml_file(get_config_path() / 'config.toml')


if __name__ == '__main__':
    ...