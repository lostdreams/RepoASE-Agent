"""
给repo agent 提供一个虚拟的运行环境


使用reposandbox对象代替repoworkspace
repo_sandbox:
    - 虚拟路径
    - 实际物理路径


每个repo都有各自的local path,也就是实际的物理地址

reposandbox支持：

- 支持运行linux命令 (使用subprocess运行,当前已确保运行环境就是在linux上)
- 包括git的命令操作
- 返回整个 sandbox的description , 包括模型可以做的工具调用操作等说明
- 支持repoworkspace的所有的操作,  

- 对于传入的repo 支持构建自己的读写等操作


"""


class Repo:
    """"
    表示一个repo对象 repo内置的所有的工具的路径都是repo内的完整相对路径
    
    """
    local_path : str
    codehub_link: str # 远程代码仓链接地址
    repo_name: str  # 默认等于local_dir下的名称
    repo_expose_funcs : literal[Repo_DeFault_Tool_list]  # 这个repo允许的操作，默认与读相关的操作都可以
    approve_git_command: bool
    approve_git_command_types : literal['git add','git reverse' ]  # 不可以改变当前的工作分支，删除历史分支 ，强制覆盖

    repo_tool_exectute_dir = '' # 所有的repo的工具的中间产物存放于此



COMMON_Linux_Command_Types = [""]


##  会保证 当前所有的repo都在repo_workspace 下

class RepoSandBox:


    def __init__(self,
        virtual_path : str = '/usr/workspace',
        disk_task : str = '/usr1/repo_workspace'
        repo_list : List[Repo],

        approved_linux_command : Literal[] 
    

    ) -> None:
        self.virtual_path = virtual_path
        self.repos = { repo.name : repo for repo in repo_list}
        self.current_virtual_
    
    
    def get_sandbox_description(self) -> str:

        return (
           f"你当前的工作路径为{self.virtual_path}",
           f"其下有{len(self.repos)} 代码仓 "
            f"其各自的含义是 "
           f" \n {repo.desriotion}"
        )

    
    # 以下是sandbox 提供的统一的能力
    def  read—（）：
        pass




    def list——dir ：
        pass


    def 
