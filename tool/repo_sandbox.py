"""
代码仓库沙箱管理模块 - 更新版

主要更新：
1. 根据repo的repo_approve_tools动态生成工具
2. 实现get_sandbox_info方法，提供详细的工具信息
3. 使用Literal类型约束repo_name参数
4. 在工具描述中明确标注可用的repo
"""

import logging
from typing import List, Dict, Any, Optional, Literal, get_args, Callable
from pathlib import Path
import inspect

from repo_agent.tool.repo import Repo
from repo_agent.core.types import FunctionTool, ToolExecutionResult

logger = logging.getLogger(__name__)


# 定义可用的工具名称
AVAILABLE_TOOL_NAMES = Literal[
    "read_repo_file",
    "create_file_in_repo", 
    "search_replace_in_repo",
    "list_repo_dir",
    "search_files_in_repo",
    "search_code_in_repo",
    "run_git_command_in_repo"
]


class RepoSandbox:
    """
    多仓库沙箱管理器
    
    为多个代码仓库提供统一的操作接口，支持基于仓库名的路由和权限控制。
    根据每个repo的repo_approve_tools动态生成可用工具，工具描述直接来自
    repo方法的docstring。
    
    Attributes:
        repos: 仓库字典，key为仓库名，value为Repo实例
        repo_names_list: 所有仓库名列表
    """
    
    def __init__(self, repos: List[Repo]) -> None:
        """
        初始化沙箱
        
        Args:
            repos: Repo实例列表
        
        Raises:
            ValueError: 当存在重名仓库时抛出
        """
        if not repos:
            raise ValueError("至少需要一个Repo实例")
        
        # 检查仓库名是否重复
        repo_names = [repo.repo_name for repo in repos]
        if len(repo_names) != len(set(repo_names)):
            duplicates = [name for name in repo_names if repo_names.count(name) > 1]
            raise ValueError(f"存在重复的仓库名: {set(duplicates)}")
        
        # 构建仓库字典
        self.repos: Dict[str, Repo] = {repo.repo_name: repo for repo in repos}
        self.repo_names_list: List[str] = sorted(repo_names)
        
        logger.info(f"初始化沙箱，包含 {len(self.repos)} 个仓库: {self.repo_names_list}")
    
    # ==================== 内部辅助方法 ====================
    
    def _get_repo(self, repo_name: str) -> Repo:
        """获取指定仓库实例"""
        if repo_name not in self.repos:
            raise ValueError(
                f"仓库 '{repo_name}' 不存在。"
                f"可用仓库: {', '.join(self.repo_names_list)}"
            )
        return self.repos[repo_name]
    
    def _validate_repo_path(self, file_path: str, repo_name: str) -> str:
        """验证并标准化文件路径"""
        if file_path.startswith(f"{repo_name}/"):
            return file_path
        
        clean_path = file_path.lstrip('./').lstrip('/')
        return f"{repo_name}/{clean_path}"
    
    def _get_repos_with_tool(self, tool_name: str) -> List[str]:
        """
        获取支持指定工具的所有仓库名称
        
        Args:
            tool_name: 工具名称（如 "read_file", "create_file"）
        
        Returns:
            支持该工具的仓库名称列表
        """
        repos_with_tool = []
        for repo_name, repo in self.repos.items():
            if tool_name in repo.repo_approve_tools:
                repos_with_tool.append(repo_name)
        return repos_with_tool
    
    def _get_method_docstring(self, repo: Repo, method_name: str) -> str:
        """
        获取repo方法的docstring
        
        Args:
            repo: Repo实例
            method_name: 方法名称
        
        Returns:
            方法的docstring，如果不存在返回空字符串
        """
        method = getattr(repo, method_name, None)
        if method and callable(method):
            return inspect.getdoc(method) or ""
        return ""
    
    def _replace_repo_placeholder(self, docstring: str, repo_names: List[str]) -> str:
        """
        替换docstring中的{repo}占位符
        
        Args:
            docstring: 原始docstring
            repo_names: 支持该工具的repo名称列表
        
        Returns:
            替换后的docstring
        """
        if not repo_names:
            return docstring
        
        # 格式化repo列表
        repos_str = ", ".join(repo_names)
        
        # 替换{repo}占位符
        result = docstring.replace("{repo}", repos_str)
        
        return result
    
    # ==================== 文件读取操作 ====================
    
    def read_repo_file(
        self,
        repo_name: str,
        repo_file_path: str
    ) -> Dict[str, Any]:
        """读取指定仓库中的文件内容"""
        try:
            repo = self._get_repo(repo_name)
            full_path = self._validate_repo_path(repo_file_path, repo_name)
            return repo.read_file(full_path)
        except ValueError as e:
            return {
                "status": "error",
                "error_type": "InvalidRepository",
                "message": str(e)
            }
        except Exception as e:
            return {
                "status": "error",
                "error_type": e.__class__.__name__,
                "message": f"读取文件时发生错误: {e}"
            }
    
    # ==================== 文件创建操作 ====================
    
    def create_file_in_repo(
        self,
        repo_name: str,
        create_file_path_in_repo: str,
        insert_content: str
    ) -> Dict[str, Any]:
        """在指定仓库中创建新文件"""
        try:
            repo = self._get_repo(repo_name)
            full_path = self._validate_repo_path(create_file_path_in_repo, repo_name)
            return repo.create_file(full_path, insert_content)
        except ValueError as e:
            return {
                "status": "error",
                "error_type": "InvalidRepository",
                "message": str(e)
            }
        except Exception as e:
            return {
                "status": "error",
                "error_type": e.__class__.__name__,
                "message": f"创建文件时发生错误: {e}"
            }
    
    # ==================== 文件搜索和替换 ====================
    
    def search_replace_in_repo(
        self,
        repo_name: str,
        file_path: str,
        search_text: str,
        replace_text: str
    ) -> Dict[str, Any]:
        """在指定仓库的文件中搜索并替换文本"""
        try:
            repo = self._get_repo(repo_name)
            full_path = self._validate_repo_path(file_path, repo_name)
            return repo.search_replace(search_text, replace_text, full_path)
        except ValueError as e:
            return {
                "status": "error",
                "error_type": "InvalidRepository",
                "message": str(e)
            }
        except Exception as e:
            return {
                "status": "error",
                "error_type": e.__class__.__name__,
                "message": f"搜索替换时发生错误: {e}"
            }
    
    # ==================== 目录和文件搜索 ====================
    
    def list_repo_dir(
        self,
        repo_name: str,
        start_dir: str = ".",
        depth: int = 2
    ) -> Dict[str, Any]:
        """列出指定仓库的目录结构"""
        try:
            repo = self._get_repo(repo_name)
            if start_dir in [".", "/", ""]:
                full_path = repo_name
            else:
                full_path = self._validate_repo_path(start_dir, repo_name)
            return repo.list_dir(full_path, depth)
        except ValueError as e:
            return {
                "status": "error",
                "error_type": "InvalidRepository",
                "message": str(e)
            }
        except Exception as e:
            return {
                "status": "error",
                "error_type": e.__class__.__name__,
                "message": f"列出目录时发生错误: {e}"
            }
    
    def search_files_in_repo(
        self,
        repo_name: str,
        query: str,
        extensions: Optional[List[str]] = None,
        limit: int = 5
    ) -> Dict[str, Any]:
        """在指定仓库中搜索文件名"""
        try:
            repo = self._get_repo(repo_name)
            return repo.search_files(query, extensions, limit)
        except ValueError as e:
            return {
                "status": "error",
                "error_type": "InvalidRepository",
                "message": str(e)
            }
        except Exception as e:
            return {
                "status": "error",
                "error_type": e.__class__.__name__,
                "message": f"搜索文件时发生错误: {e}"
            }
    
    def search_code_in_repo(
        self,
        repo_name: str,
        query: str,
        files_to_search: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """在指定仓库的代码中搜索文本"""
        try:
            repo = self._get_repo(repo_name)
            
            if files_to_search:
                files_to_search = [
                    self._validate_repo_path(f, repo_name) for f in files_to_search
                ]
            
            return repo.search_code(query, files_to_search)
        except ValueError as e:
            return {
                "status": "error",
                "error_type": "InvalidRepository",
                "message": str(e)
            }
        except Exception as e:
            return {
                "status": "error",
                "error_type": e.__class__.__name__,
                "message": f"搜索代码时发生错误: {e}"
            }
    
    # ==================== Git操作 ====================
    
    def run_git_command_in_repo(
        self,
        repo_name: str,
        command: str
    ) -> Dict[str, Any]:
        """在指定仓库中执行Git命令"""
        try:
            repo = self._get_repo(repo_name)
            return repo.run_git_command(command)
        except ValueError as e:
            return {
                "status": "error",
                "error_type": "InvalidRepository",
                "message": str(e)
            }
        except Exception as e:
            return {
                "status": "error",
                "error_type": e.__class__.__name__,
                "message": f"执行Git命令时发生错误: {e}"
            }
    
    # ==================== 工具接口生成 ====================
    
    def get_sandbox_tools(
        self,
        tool_names: Optional[List[str]] = None
    ) -> List[FunctionTool]:
        """
        获取沙箱提供的工具函数列表
        
        工具描述直接来自repo方法的docstring，并将{repo}占位符替换为
        实际支持该工具的repo名称列表。repo_name参数使用Literal类型约束。
        
        Args:
            tool_names: 需要的工具名称列表，为None时返回所有可用工具
                       可选值: "read_repo_file", "create_file_in_repo", 
                              "search_replace_in_repo", "list_repo_dir",
                              "search_files_in_repo", "search_code_in_repo",
                              "run_git_command_in_repo"
        
        Returns:
            FunctionTool列表
        """
        # 定义工具映射：沙箱工具名 -> (沙箱方法, repo方法名, 参数映射)
        tool_map = {
            "read_repo_file": {
                "sandbox_method": self.read_repo_file,
                "repo_method_name": "read_file",
                "param_mapping": {"repo_file_path": "file_path"}
            },
            "list_repo_dir": {
                "sandbox_method": self.list_repo_dir,
                "repo_method_name": "list_dir",
                "param_mapping": {"start_dir": "start_dir", "depth": "depth"}
            },
            "search_files_in_repo": {
                "sandbox_method": self.search_files_in_repo,
                "repo_method_name": "search_files",
                "param_mapping": {"query": "query", "extensions": "extensions", "limit": "limit"}
            },
            "search_code_in_repo": {
                "sandbox_method": self.search_code_in_repo,
                "repo_method_name": "search_code",
                "param_mapping": {"query": "query", "files_to_search": "files_to_search"}
            },
            "create_file_in_repo": {
                "sandbox_method": self.create_file_in_repo,
                "repo_method_name": "create_file",
                "param_mapping": {"create_file_path_in_repo": "file_path", "insert_content": "file_content"}
            },
            "search_replace_in_repo": {
                "sandbox_method": self.search_replace_in_repo,
                "repo_method_name": "search_replace",
                "param_mapping": {"file_path": "file_path", "search_text": "search_text", "replace_text": "replace_text"}
            },
            "run_git_command_in_repo": {
                "sandbox_method": self.run_git_command_in_repo,
                "repo_method_name": "run_git_command",
                "param_mapping": {"command": "command"}
            }
        }
        
        # 如果未指定工具列表，返回所有至少有一个repo支持的工具
        if tool_names is None:
            tool_names = []
            for tool_name, tool_info in tool_map.items():
                repos_with_tool = self._get_repos_with_tool(tool_info["repo_method_name"])
                if repos_with_tool:
                    tool_names.append(tool_name)
        
        # 构建 FunctionTool 列表
        result = []
        for tool_name in tool_names:
            if tool_name not in tool_map:
                available = ", ".join(tool_map.keys())
                logger.warning(f"未知的工具名称: {tool_name}。可用工具: {available}")
                continue
            
            tool_info = tool_map[tool_name]
            repo_method_name = tool_info["repo_method_name"]
            sandbox_method = tool_info["sandbox_method"]
            
            # 获取支持该工具的repo列表
            repos_with_tool = self._get_repos_with_tool(repo_method_name)
            
            if not repos_with_tool:
                logger.info(f"跳过工具 {tool_name}：没有仓库支持 {repo_method_name} 操作")
                continue
            
            # 获取原始方法的docstring
            first_repo = self.repos[repos_with_tool[0]]
            original_docstring = self._get_method_docstring(first_repo, repo_method_name)
            
            # 替换{repo}占位符
            final_docstring = self._replace_repo_placeholder(original_docstring, repos_with_tool)
            
            # 创建带Literal类型约束的wrapper函数
            wrapper_func = self._create_typed_wrapper(
                tool_name=tool_name,
                sandbox_method=sandbox_method,
                repos_with_tool=repos_with_tool,
                docstring=final_docstring
            )
            
            # 创建FunctionTool
            result.append(FunctionTool(
                func=wrapper_func,
                name=tool_name,
                description=final_docstring
            ))
        
        return result
    
    def _create_typed_wrapper(
        self,
        tool_name: str,
        sandbox_method: Callable,
        repos_with_tool: List[str],
        docstring: str
    ) -> Callable:
        """
        创建带Literal类型注解的wrapper函数
        
        重要说明：
        - 绑定方法（self.method）的签名中已经不包含 self
        - 第一个参数直接就是目标参数（如 repo_name）
        - 不需要用 params[1:] 跳过任何参数
        
        Args:
            tool_name: 工具名称
            sandbox_method: 沙箱绑定方法（如self.read_repo_file）
            repos_with_tool: 支持该工具的repo名称列表
            docstring: 处理后的docstring
        
        Returns:
            wrapper函数，第一个参数带有Literal类型约束
        """
        # 获取绑定方法的签名（注意：没有 self 参数）
        sig = inspect.signature(sandbox_method)
        params = list(sig.parameters.values())
        
        # ✅ 修复：直接处理所有参数，不要跳过第一个
        # 创建新的参数列表
        # 第一个参数添加Literal类型约束，其他参数保持不变
        new_params = []
        for i, param in enumerate(params):  # ✅ 正确！不跳过任何参数
            if i == 0:
                # 第一个参数（repo_name）添加Literal类型约束
                RepoNameLiteral = Literal[tuple(repos_with_tool)]  # type: ignore
                new_param = param.replace(annotation=RepoNameLiteral)
                new_params.append(new_param)
            else:
                # 其他参数保持原样
                new_params.append(param)
        
        # 创建wrapper函数
        def wrapper(*args, **kwargs):
            return sandbox_method(*args, **kwargs)
        
        # 设置wrapper的元数据
        wrapper.__name__ = tool_name
        wrapper.__doc__ = docstring
        wrapper.__signature__ = sig.replace(parameters=new_params)  # type: ignore
        
        # 设置类型注解（FunctionTool需要）
        annotations = {}
        for param in new_params:
            if param.annotation != inspect.Parameter.empty:
                annotations[param.name] = param.annotation
        annotations['return'] = Dict[str, Any]
        wrapper.__annotations__ = annotations
        
        return wrapper
        
    # ==================== 子沙箱创建 ====================
    
    def get_subsandbox(
        self,
        sub_repo_names: List[str]
    ) -> "RepoSandbox":
        """创建子沙箱，包含当前沙箱中的部分仓库"""
        invalid_names = [name for name in sub_repo_names if name not in self.repos]
        if invalid_names:
            raise ValueError(
                f"以下仓库不存在: {', '.join(invalid_names)}。"
                f"可用仓库: {', '.join(self.repo_names_list)}"
            )
        
        sub_repos = [self.repos[name] for name in sub_repo_names]
        return RepoSandbox(sub_repos)
    
    # ==================== 实用方法 ====================
    
    def get_sandbox_info(self) -> str:
        """
        获取沙箱的详细信息，包括可用仓库和工具权限
        
        Returns:
            格式化的沙箱信息字符串
        """
        lines = []
    #    lines.append("=" * 80)
        lines.append("当前任务可用代码仓信息")
    #    lines.append("=" * 80)
        lines.append("")
        
        # 1. 可用代码仓列表
        lines.append("## 可用代码仓")
        lines.append(f"共 {len(self.repos)} 个代码仓:")
        for repo_name in self.repo_names_list:
            repo = self.repos[repo_name]
            lines.append(f"  - 代码仓名称 {repo_name}")
            
        #    lines.append(f"    路径: {repo.local_path}")
            if repo.codehub_link:
                lines.append(f"    远程链接: {repo.codehub_link}")
            lines.append(f"    文件数: {len(repo.all_files)}")
        lines.append("")
        
        # 2. 读操作工具
        lines.append("## 读操作工具")
        lines.append("以下工具可用于读取代码仓信息:")
        
        read_tools = {
            "read_file": "读取文件内容",
            "list_dir": "列出目录结构",
            "search_files": "搜索文件名",
            "search_code": "搜索代码内容"
        }
        
        for tool_name, tool_desc in read_tools.items():
            repos_with_tool = self._get_repos_with_tool(tool_name)
            if repos_with_tool:
                lines.append(f"  - {tool_name}: {tool_desc}")
                lines.append(f"    支持的仓库: {', '.join(repos_with_tool)}")
        lines.append("")
        
        # 3. 写操作工具
        lines.append("## 写操作工具")
        lines.append("以下仓库支持写操作:")
        
        write_tools = {
            "create_file": "创建新文件",
            "search_replace": "搜索并替换内容"
        }
        
        has_write_repo = False
        for repo_name, repo in self.repos.items():
            write_perms = []
            for tool_name, tool_desc in write_tools.items():
                if tool_name in repo.repo_approve_tools:
                    write_perms.append(f"{tool_name}({tool_desc})")
            
            if write_perms:
                has_write_repo = True
                lines.append(f"  - {repo_name}:")
                for perm in write_perms:
                    lines.append(f"      {perm}")
        
        if not has_write_repo:
            lines.append("  (无仓库支持写操作)")
        lines.append("")
        
        # 4. Git命令工具
        lines.append("## Git命令工具")
        lines.append("以下仓库支持Git命令:")
        
        has_git_repo = False
        for repo_name, repo in self.repos.items():
            if "run_git_command" in repo.repo_approve_tools:
                has_git_repo = True
                lines.append(f"  - {repo_name}:")
                lines.append(f"    允许的命令: {', '.join(repo.approve_git_command_types)}")
        
        if not has_git_repo:
            lines.append(" 当前所有代码仓都不支持写入修改的工具执行")
        lines.append("")
        
        # 5. 使用示例
        lines.append("## 工具调用示例")
        lines.append("```python")
        if self.repo_names_list:
            example_repo = self.repo_names_list[0]
            lines.append(f'# 读取文件')
            lines.append(f'sandbox.read_repo_file("{example_repo}", "{example_repo}/src/main.py")')
        
        write_repos = [name for name, repo in self.repos.items() if "create_file" in repo.repo_approve_tools]
        if write_repos:
            lines.append("")
            lines.append("# 创建文件（仅限支持写操作的仓库）")
            lines.append(f'sandbox.create_file_in_repo("{write_repos[0]}", "{write_repos[0]}/new_file.py", "# content")')
        
        git_repos = [name for name, repo in self.repos.items() if "run_git_command" in repo.repo_approve_tools]
        if git_repos:
            lines.append("")
            lines.append("# Git命令（仅限支持Git的仓库）")
            lines.append(f'sandbox.run_git_command_in_repo("{git_repos[0]}", "git status")')
        
        lines.append("```")
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def get_repo_info(self) -> Dict[str, Any]:
        """
        获取沙箱中所有仓库的信息（结构化数据）
        
        Returns:
            包含仓库信息的字典
        """
        info = {
            "total_repos": len(self.repos),
            "repos": []
        }
        
        for repo_name in self.repo_names_list:
            repo = self.repos[repo_name]
            info["repos"].append({
                "name": repo.repo_name,
                "path": repo.local_path,
                "description": repo.description,
                "file_count": len(repo.all_files),
                "available_operations": repo.repo_approve_tools,
                "git_enabled": "run_git_command" in repo.repo_approve_tools,
                "git_commands": repo.approve_git_command_types if "run_git_command" in repo.repo_approve_tools else []
            })
        
        return info
    
    def __repr__(self) -> str:
        """返回沙箱的字符串表示"""
        return f"RepoSandbox(repos={self.repo_names_list})"
