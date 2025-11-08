"""
代码仓库工作空间管理模块

主要功能：
- 多仓库统一管理
- 基于路径的自动路由
- 工具函数的权限控制
- 子仓库工具生成

核心类：
- Repo: 单个代码仓库的封装，提供基础文件操作
- RepoWorkSpace: 多仓库工作空间，提供统一的工具接口和路由功能
"""

import os
import re
import pickle
import fnmatch
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from thefuzz import process as fuzzy_process
import json
import logging

from repo_agent.core.types import FunctionTool

logger = logging.getLogger(__name__)

# Repo默认提供的工具列表
Repo_Default_Tool_List = [
    "search_files",
    "search_code",
    "list_dir",
    "read_file",
]

Repo_Write_Tool_list = [
    "create_file",
    "search_replace",
]

# Git相关的安全操作列表
Safe_Git_Commands = [
    "git status",
    "git log",
    "git diff",
    "git show",
    "git branch",
    "git add",
    "git commit",
]


class Repo:
    """
    单个代码仓库对象
    
    提供文件读写、搜索、目录浏览等基础操作。
    所有路径格式为：{repo_name}/path/to/file
    
    Attributes:
        repo_dir: 仓库本地绝对路径
        repo_name: 仓库名称
        local_path: 仓库本地路径字符串
        codehub_link: 远程仓库链接
        allowed_operations: 允许的操作列表
        all_files: 扫描到的所有文件列表
    """
    
    def __init__(
        self,
        local_path: str | Path,
        repo_name: Optional[str] = None,
        codehub_link: Optional[str] = None,
        repo_ignore_patterns: Optional[List[str]] = None,
        scan_patterns: List[str] = ['**/*'],
        repo_expose_funcs: List[str] | None = None,
        approve_git_command: bool = False,
        approve_git_command_types: List[str] | None = None,
        cache_dir: Optional[str | Path] = None,
        repo_description: Optional[str] = None,
    ):
        # 基础路径设置
        self.repo_dir = Path(local_path).resolve()
        if not self.repo_dir.is_dir():
            raise FileNotFoundError(f"仓库目录不存在: {self.repo_dir}")
        
        self.repo_name = repo_name or self.repo_dir.name
        self.local_path = str(self.repo_dir)
        self.codehub_link = codehub_link
        self._description = repo_description
        
        # 扫描和忽略模式
        self.repo_ignore_patterns = repo_ignore_patterns or []
        self.scan_patterns = scan_patterns
        
        # 权限设置
        self.allowed_operations = repo_expose_funcs or Repo_Default_Tool_List.copy()
        self.approve_git_command = approve_git_command
        self.approve_git_command_types = approve_git_command_types or Safe_Git_Commands.copy()
        
        # 缓存目录
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / '.repo_analyzer_cache'
        
        self.repo_cache_dir = self.cache_dir / self.repo_name
        self.repo_tool_execute_dir = self.repo_cache_dir / 'execute'
        self.scan_cache_path = self.repo_cache_dir / 'scan.cache'
        
        # 创建必要的目录
        self.repo_cache_dir.mkdir(parents=True, exist_ok=True)
        self.repo_tool_execute_dir.mkdir(parents=True, exist_ok=True)
        
        # 文件列表
        self.all_files: List[str] = []
        self._is_scanned = False
        
        # 扫描文件
        self._ensure_files_scanned()
    
    @property
    def description(self) -> str:
        """返回仓库描述"""
        return self._description or f"代码仓库: {self.repo_name}"
    
    # ==================== 内部辅助方法 ====================
    
    def _ensure_files_scanned(self):
        """确保仓库文件已被扫描"""
        if self._is_scanned:
            return
        
        if self._load_from_cache():
            logger.info(f"从缓存加载仓库 '{self.repo_name}' 的文件列表")
        else:
            logger.info(f"扫描仓库 '{self.repo_name}'...")
            self._scan_repo_files()
            self._save_to_cache()
        
        self._is_scanned = True
    
    def _to_relative_str(self, absolute_path: Path) -> str:
        """将绝对路径转换为包含仓库名的相对路径字符串"""
        rel_path = absolute_path.relative_to(self.repo_dir)
        return f"{self.repo_name}/{str(rel_path).replace(os.sep, '/')}"
    
    def _to_absolute_path(self, relative_path_str: str) -> Path:
        """将包含仓库名的相对路径字符串转换为绝对路径"""
        # 移除开头的仓库名
        if relative_path_str.startswith(f"{self.repo_name}/"):
            relative_path_str = relative_path_str[len(self.repo_name) + 1:]
        
        clean_path = relative_path_str.lstrip('./').lstrip('/')
        return self.repo_dir / clean_path.replace('\\', '/')
    
    def _is_ignored(self, relative_path_str: str) -> bool:
        """检查路径是否匹配忽略模式"""
        path_to_check = relative_path_str.replace('\\', '/')
        return any(fnmatch.fnmatch(path_to_check, pattern) for pattern in self.repo_ignore_patterns)
    
    def _scan_repo_files(self):
        """扫描仓库文件"""
        all_files_set = set()
        
        for root, dirnames, filenames in os.walk(self.repo_dir, topdown=True):
            current_root = Path(root)
            relative_root = self._to_relative_str(current_root)
            
            # 过滤忽略的目录
            dirnames[:] = [
                d for d in dirnames 
                if not self._is_ignored(f"{relative_root}/{d}".lstrip('/'))
            ]
            
            for filename in filenames:
                relative_file = f"{relative_root}/{filename}".lstrip('/')
                
                if self._is_ignored(relative_file):
                    continue
                
                if any(fnmatch.fnmatch(relative_file, pattern) for pattern in self.scan_patterns):
                    all_files_set.add(relative_file)
        
        self.all_files = sorted(list(all_files_set))
        logger.info(f"扫描完成：找到 {len(self.all_files)} 个文件")
    
    def _load_from_cache(self) -> bool:
        """从缓存加载文件列表"""
        if not self.scan_cache_path.exists():
            return False
        
        try:
            with open(self.scan_cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            
            if 'all_files_relative' in cached_data and isinstance(cached_data['all_files_relative'], list):
                self.all_files = cached_data['all_files_relative']
                return True
            return False
        except Exception as e:
            logger.warning(f"加载缓存失败: {e}")
            return False
    
    def _save_to_cache(self):
        """保存文件列表到缓存"""
        try:
            cache_data = {'all_files_relative': self.all_files}
            with open(self.scan_cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"缓存已保存: {self.scan_cache_path}")
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")
    
    def _check_permission(self, operation: str) -> Optional[Dict[str, Any]]:
        """
        检查操作权限
        
        Returns:
            None 表示允许，Dict 表示拒绝并包含错误信息
        """
        if operation not in self.allowed_operations:
            return {
                "status": "error",
                "error_type": "PermissionError",
                "message": f"当前仓库不允许 {operation} 操作"
            }
        return None
    
    # ==================== 公共API方法 ====================
    
    def read_file(self, file_path: str) -> Dict[str, Any]:
        """
        读取文件内容
        
        Args:
            file_path: 文件路径，格式为 {repo_name}/path/to/file
        
        Returns:
            {"status": "success", "data": content} 或错误信息
        """
        if err := self._check_permission("read_file"):
            return err
        
        if file_path not in self.all_files:
            matches = fuzzy_process.extract(file_path, self.all_files, limit=3)
            suggestions = [match[0] for match in matches if match[1] > 70]
            return {
                "status": "error",
                "error_type": "FileNotFoundError",
                "message": f"文件 '{file_path}' 不存在",
                "suggestions": suggestions
            }
        
        try:
            target_path = self._to_absolute_path(file_path)
            content = target_path.read_text(encoding='utf-8', errors='ignore')
            return {"status": "success", "data": content}
        except Exception as e:
            return {
                "status": "error",
                "error_type": e.__class__.__name__,
                "message": f"读取文件失败: {e}"
            }
    
    def list_dir(self, start_dir: str, depth: int = 2) -> Dict[str, Any]:
        """
        列出目录结构
        
        Args:
            start_dir: 起始目录，格式为 {repo_name}/path 或 '.' 表示根目录
            depth: 显示深度
        
        Returns:
            {"status": "success", "data": tree_structure} 或错误信息
        """
        if err := self._check_permission("list_dir"):
            return err
        
        # 处理根目录的各种表示
        if start_dir in ['.', '/', self.repo_name, f'{self.repo_name}/']:
            target_dir = self.repo_name
        else:
            target_dir = start_dir
        
        # 确保路径以仓库名开头
        if not (target_dir == self.repo_name or target_dir.startswith(f"{self.repo_name}/")):
            return {
                "status": "error",
                "error_type": "InvalidPathError",
                "message": f"路径必须以 '{self.repo_name}' 开头"
            }
        
        start_path = self._to_absolute_path(target_dir)
        
        if not start_path.is_dir():
            return {
                "status": "error",
                "error_type": "DirectoryNotFoundError",
                "message": f"目录 '{target_dir}' 不存在"
            }
        
        header = f"'{start_path.name}' [depth={depth}]"
        tree_lines = self._build_tree_lines(start_path, prefix='', depth=depth)
        full_tree = "\n".join([header] + tree_lines)
        
        return {"status": "success", "data": full_tree}
    
    def _build_tree_lines(self, dir_path: Path, prefix: str = '', depth: int = 2) -> List[str]:
        """递归构建目录树"""
        if depth < 0:
            return []
        
        lines = []
        try:
            items = sorted(
                [item for item in dir_path.iterdir() if not self._is_ignored(self._to_relative_str(item))],
                key=lambda p: (p.is_file(), p.name.lower())
            )
        except OSError as e:
            lines.append(f"{prefix}└── [无法访问: {e.strerror}]")
            return lines
        
        for i, item in enumerate(items):
            connector = '└── ' if i == len(items) - 1 else '├── '
            lines.append(f"{prefix}{connector}{item.name}")
            
            if item.is_dir():
                new_prefix = prefix + ('    ' if i == len(items) - 1 else '│   ')
                lines.extend(self._build_tree_lines(item, new_prefix, depth - 1))
        
        return lines
    
    def search_files(self, query: str, extensions: Optional[List[str]] = None, limit: int = 5) -> Dict[str, Any]:
        """
        搜索文件名
        
        Args:
            query: 搜索关键词
            extensions: 文件扩展名过滤列表，如 ['.py', '.md']
            limit: 返回结果数量限制
        
        Returns:
            {"status": "success", "data": [file_paths]} 或错误信息
        """
        if err := self._check_permission("search_files"):
            return err
        
        matches = fuzzy_process.extract(query, self.all_files, limit=limit * 2)
        suggestions = [match[0] for match in matches if match[1] > 60]
        
        if extensions:
            normalized_exts = [ext if ext.startswith('.') else f'.{ext}' for ext in extensions]
            suggestions = [path for path in suggestions if any(path.endswith(ext) for ext in normalized_exts)]
        
        suggestions = suggestions[:limit]
        
        if suggestions:
            return {"status": "success", "data": suggestions}
        else:
            return {"status": "error", "error_type": "NotFound", "message": f"未找到匹配 '{query}' 的文件"}
    
    def search_code(self, query: str, files_to_search: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        在文件内容中搜索文本
        
        Args:
            query: 搜索文本
            files_to_search: 限定搜索的文件列表，为None时搜索所有文件
        
        Returns:
            {"status": "success", "data": [usage_info]} 或错误信息
        """
        if err := self._check_permission("search_code"):
            return err
        
        try:
            query_regex = re.compile(r'\b' + re.escape(query) + r'\b')
        except re.error as e:
            return {"status": "error", "error_type": "InvalidQuery", "message": f"无效的搜索查询: {e}"}
        
        files = files_to_search if files_to_search else self.all_files
        usages = []
        
        for file_path in files:
            absolute_path = self._to_absolute_path(file_path)
            
            if not absolute_path.is_file():
                continue
            
            try:
                with open(absolute_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f, 1):
                        if query_regex.search(line):
                            usages.append({
                                "file_path": file_path,
                                "line_number": line_num,
                                "line_content": line.strip()
                            })
            except (IOError, OSError):
                continue
        
        if usages:
            return {"status": "success", "data": usages}
        else:
            return {"status": "error", "error_type": "NotFound", "message": f"未找到 '{query}'"}
    
    def create_file(self, file_path: str, file_content: str = "") -> Dict[str, Any]:
        """
        创建新文件
        
        Args:
            file_path: 文件路径，格式为 {repo_name}/path/to/file
            file_content: 文件内容
        
        Returns:
            操作结果
        """
        if err := self._check_permission("create_file"):
            return err
        
        if not file_path.startswith(f"{self.repo_name}/"):
            return {
                "status": "error",
                "error_type": "InvalidPathError",
                "message": f"路径必须以 '{self.repo_name}/' 开头"
            }
        
        try:
            absolute_path = self._to_absolute_path(file_path)
            
            if absolute_path.exists():
                return {
                    "status": "error",
                    "error_type": "FileExistsError",
                    "message": f"文件 '{file_path}' 已存在"
                }
            
            absolute_path.parent.mkdir(parents=True, exist_ok=True)
            absolute_path.write_text(file_content, encoding='utf-8')
            
            # 更新文件列表
            self.all_files.append(file_path)
            self.all_files.sort()
            self._save_to_cache()
            
            return {"status": "success", "message": f"文件 '{file_path}' 创建成功"}
        except Exception as e:
            return {
                "status": "error",
                "error_type": e.__class__.__name__,
                "message": f"创建文件失败: {e}"
            }
    
    def search_replace(self, search_text: str, replace_text: str, file_path: str) -> Dict[str, Any]:
        """
        搜索并替换文件内容
        
        Args:
            search_text: 要搜索的文本
            replace_text: 替换后的文本
            file_path: 目标文件路径
        
        Returns:
            操作结果
        """
        if err := self._check_permission("search_replace"):
            return err
        
        if not search_text:
            return {"status": "error", "error_type": "InvalidQuery", "message": "搜索文本不能为空"}
        
        if file_path not in self.all_files:
            return {
                "status": "error",
                "error_type": "FileNotFoundError",
                "message": f"文件 '{file_path}' 不存在"
            }
        
        try:
            absolute_path = self._to_absolute_path(file_path)
            
            if not absolute_path.is_file():
                return {
                    "status": "error",
                    "error_type": "FileNotFoundError",
                    "message": f"文件 '{file_path}' 不存在于文件系统"
                }
            
            original_content = absolute_path.read_text(encoding='utf-8', errors='ignore')
            
            if search_text not in original_content:
                return {
                    "status": "success",
                    "message": f"在文件中未找到 '{search_text}'",
                    "data": {"modified": False}
                }
            
            new_content = original_content.replace(search_text, replace_text)
            absolute_path.write_text(new_content, encoding='utf-8')
            
            return {
                "status": "success",
                "message": f"成功替换 '{search_text}'",
                "data": {"modified": True}
            }
        except Exception as e:
            return {
                "status": "error",
                "error_type": e.__class__.__name__,
                "message": f"操作失败: {e}"
            }


class RepoWorkSpace:
    """
    多仓库工作空间管理器
    
    提供统一的多仓库管理接口，支持：
    - 自动路由：根据路径前缀自动路由到对应仓库
    - 统一工具：将多个仓库的操作封装为统一的工具接口
    - 子空间：支持创建特定路径的子工作空间工具
    
    所有操作的路径格式为：{repo_name}/path/to/file
    """
    
    def __init__(self, repos: List[Repo], workspace_name: str = "default_workspace"):
        """
        初始化工作空间
        
        Args:
            repos: 要添加到工作空间的Repo实例列表
            workspace_name: 工作空间名称
        """
        self.workspace_name = workspace_name
        self.repos: Dict[str, Repo] = {}
        
        for repo in repos:
            self.add_repo(repo)
        
        logger.info(f"工作空间 '{workspace_name}' 初始化完成，包含 {len(self.repos)} 个仓库")
    
    def add_repo(self, repo: Repo) -> None:
        """添加仓库到工作空间"""
        if repo.repo_name in self.repos:
            logger.warning(f"仓库 '{repo.repo_name}' 已存在，将被覆盖")
        self.repos[repo.repo_name] = repo
        logger.info(f"添加仓库: {repo.repo_name}")
    
    def remove_repo(self, repo_name: str) -> bool:
        """从工作空间移除仓库"""
        if repo_name in self.repos:
            del self.repos[repo_name]
            logger.info(f"移除仓库: {repo_name}")
            return True
        return False
    
    def _parse_path(self, file_path: str) -> tuple[Optional[str], str]:
        """
        解析路径，提取仓库名和相对路径
        
        Args:
            file_path: 完整路径，格式为 {repo_name}/path/to/file
        
        Returns:
            (repo_name, file_path) 元组
        """
        parts = file_path.split('/', 1)
        if len(parts) < 1:
            return None, file_path
        
        repo_name = parts[0]
        return repo_name if repo_name in self.repos else None, file_path
    
    def _route(self, file_path: str) -> Optional[Repo]:
        """根据路径路由到对应的仓库"""
        repo_name, _ = self._parse_path(file_path)
        return self.repos.get(repo_name)
    
    # ==================== 统一的工作空间操作方法 ====================
    
    def read_file(self, file_path: str) -> Dict[str, Any]:
        """
        读取文件内容（自动路由）
        
        Args:
            file_path: 文件路径，格式为 {repo_name}/path/to/file
        """
        repo = self._route(file_path)
        if not repo:
            return {
                "status": "error",
                "error_type": "RepoNotFoundError",
                "message": f"无法找到路径 '{file_path}' 对应的仓库"
            }
        
        return repo.read_file(file_path)
    
    def list_dir(self, start_dir: str, depth: int = 2) -> Dict[str, Any]:
        """列出目录结构（自动路由）"""
        repo = self._route(start_dir)
        if not repo:
            return {
                "status": "error",
                "error_type": "RepoNotFoundError",
                "message": f"无法找到路径 '{start_dir}' 对应的仓库"
            }
        
        return repo.list_dir(start_dir, depth)
    
    def search_files(self, query: str, repo_name: Optional[str] = None, 
                    extensions: Optional[List[str]] = None, limit: int = 5) -> Dict[str, Any]:
        """
        搜索文件名
        
        Args:
            query: 搜索关键词
            repo_name: 指定仓库名，为None时搜索所有仓库
            extensions: 文件扩展名过滤
            limit: 每个仓库返回的结果数量
        """
        if repo_name:
            repo = self.repos.get(repo_name)
            if not repo:
                return {
                    "status": "error",
                    "error_type": "RepoNotFoundError",
                    "message": f"仓库 '{repo_name}' 不存在"
                }
            return repo.search_files(query, extensions, limit)
        
        # 搜索所有仓库
        all_results = []
        for repo in self.repos.values():
            result = repo.search_files(query, extensions, limit)
            if result["status"] == "success":
                all_results.extend(result["data"])
        
        if all_results:
            return {"status": "success", "data": all_results[:limit * len(self.repos)]}
        else:
            return {"status": "error", "error_type": "NotFound", "message": f"未找到匹配 '{query}' 的文件"}
    
    def search_code(self, query: str, repo_name: Optional[str] = None,
                   files_to_search: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        搜索代码内容
        
        Args:
            query: 搜索文本
            repo_name: 指定仓库名，为None时搜索所有仓库
            files_to_search: 限定搜索的文件列表
        """
        if repo_name:
            repo = self.repos.get(repo_name)
            if not repo:
                return {
                    "status": "error",
                    "error_type": "RepoNotFoundError",
                    "message": f"仓库 '{repo_name}' 不存在"
                }
            return repo.search_code(query, files_to_search)
        
        # 搜索所有仓库
        all_usages = []
        for repo in self.repos.values():
            result = repo.search_code(query, files_to_search)
            if result["status"] == "success":
                all_usages.extend(result["data"])
        
        if all_usages:
            return {"status": "success", "data": all_usages}
        else:
            return {"status": "error", "error_type": "NotFound", "message": f"未找到 '{query}'"}
    
    def create_file(self, file_path: str, file_content: str = "") -> Dict[str, Any]:
        """创建文件（自动路由）"""
        repo = self._route(file_path)
        if not repo:
            return {
                "status": "error",
                "error_type": "RepoNotFoundError",
                "message": f"无法找到路径 '{file_path}' 对应的仓库"
            }
        
        return repo.create_file(file_path, file_content)
    
    def search_replace(self, search_text: str, replace_text: str, file_path: str) -> Dict[str, Any]:
        """搜索替换（自动路由）"""
        repo = self._route(file_path)
        if not repo:
            return {
                "status": "error",
                "error_type": "RepoNotFoundError",
                "message": f"无法找到路径 '{file_path}' 对应的仓库"
            }
        
        return repo.search_replace(search_text, replace_text, file_path)
    
    # ==================== 工具生成方法 ====================
    
    def get_workspace_tools(self) -> List[FunctionTool]:
        """
        获取工作空间的所有工具
        
        返回FunctionTool列表，直接绑定方法无需wrapper
        """
        return [
            FunctionTool(
                func=self.read_file,
                name="read_file",
                description="读取仓库中的文件内容，路径格式: {repo_name}/path/to/file"
            ),
            FunctionTool(
                func=self.list_dir,
                name="list_dir",
                description="以树状结构列出目录内容，路径格式: {repo_name}/path"
            ),
            FunctionTool(
                func=self.search_files,
                name="search_files",
                description="通过文件名搜索文件，可指定仓库名或搜索所有仓库"
            ),
            FunctionTool(
                func=self.search_code,
                name="search_code",
                description="在文件内容中搜索指定文本"
            ),
            FunctionTool(
                func=self.create_file,
                name="create_file",
                description="在仓库中创建新文件"
            ),
            FunctionTool(
                func=self.search_replace,
                name="search_replace",
                description="在指定文件中搜索并替换文本"
            ),
        ]
    
    def get_subrepo_tools(self, subrepo_path: str) -> List[FunctionTool]:
        """
        获取限定在子目录下的工具
        
        核心思路：
        1. 通用的路径验证函数
        2. 闭包捕获 subrepo_path 和对应的 repo 方法
        3. 所有工具共享同一个验证逻辑
        
        Args:
            subrepo_path: 子仓库路径，格式: {repo_name}/path/to/subdir
        
        Returns:
            路径受限的工具列表
        
        Example:
            >>> tools = workspace.get_subrepo_tools("linux_kernel/drivers/net")
            >>> # read_file("tcp.c") 自动补全为 linux_kernel/drivers/net/tcp.c
        """
        # 1. 验证路径并获取对应 repo
        repo = self._route(subrepo_path)
        if not repo:
            logger.error(f"无法找到路径对应的仓库: {subrepo_path}")
            return []
        
        abs_path = repo._to_absolute_path(subrepo_path)
        if not abs_path.is_dir():
            logger.error(f"路径不是目录: {subrepo_path}")
            return []
        
        # 2. 通用的路径验证和调用函数
        def _create_restricted_func(repo_method: Callable, path_param_name: str) -> Callable:
            """
            创建路径受限的函数
            
            Args:
                repo_method: repo 的原始方法（如 repo.read_file）
                path_param_name: 路径参数的名称（如 "file_path"）
            
            Returns:
                带路径验证的包装函数
            """
            def restricted_func(**kwargs) -> Dict[str, Any]:
                # 提取相对路径参数
                relative_path = kwargs.get(path_param_name)
                
                if not relative_path:
                    return {
                        "status": "error",
                        "error_type": "InvalidArgument",
                        "message": f"缺少参数: {path_param_name}"
                    }
                
                # 处理特殊情况："." 表示子仓库根目录
                if relative_path == ".":
                    full_path = subrepo_path
                else:
                    # 自动补全为完整路径
                    full_path = f"{subrepo_path}/{relative_path}".replace("//", "/")
                
                # 安全检查：确保路径在子仓库内
                if not (full_path.startswith(subrepo_path + "/") or full_path == subrepo_path):
                    return {
                        "status": "error",
                        "error_type": "SecurityError",
                        "message": f"路径越界: '{relative_path}' 必须在 '{subrepo_path}' 内"
                    }
                
                # 替换为完整路径并调用原方法
                kwargs[path_param_name] = full_path
                return repo_method(**kwargs)
            
            return restricted_func
        
        # 3. 工具配置表（定义哪些方法需要路径验证）
        tool_configs = [
            {
                "name": "read_file",
                "method": repo.read_file,
                "path_param": "file_path",
                "description": f"读取 {subrepo_path} 目录下的文件，传入相对路径（如 'tcp.c'）"
            },
            {
                "name": "list_dir",
                "method": repo.list_dir,
                "path_param": "start_dir",
                "description": f"列出 {subrepo_path} 目录结构，传入相对路径或 '.' 表示根目录"
            },
            {
                "name": "create_file",
                "method": repo.create_file,
                "path_param": "file_path",
                "description": f"在 {subrepo_path} 中创建文件"
            },
            {
                "name": "search_replace",
                "method": repo.search_replace,
                "path_param": "file_path",
                "description": f"在 {subrepo_path} 的文件中替换内容"
            },
        ]
        
        # 4. 批量创建工具
        tools = []
        for config in tool_configs:
            restricted_func = _create_restricted_func(
                repo_method=config["method"],
                path_param_name=config["path_param"]
            )
            
            tools.append(FunctionTool(
                func=restricted_func,
                name=config["name"],
                description=config["description"]
            ))
        
        # 5. 处理特殊工具（不需要路径参数的）
        
        # search_files - 需要过滤结果
        def search_files_in_subrepo(query: str, extensions: Optional[List[str]] = None, limit: int = 5) -> Dict[str, Any]:
            """在子仓库中搜索文件名"""
            result = repo.search_files(query, extensions, limit * 2)
            
            if result["status"] != "success":
                return result
            
            # 过滤：只保留子仓库内的文件
            filtered = [
                path for path in result["data"]
                if path.startswith(subrepo_path)
            ][:limit]
            
            return {"status": "success", "data": filtered} if filtered else {
                "status": "error",
                "error_type": "NotFound",
                "message": f"在 {subrepo_path} 中未找到匹配文件"
            }
        
        tools.append(FunctionTool(
            func=search_files_in_subrepo,
            name="search_files",
            description=f"在 {subrepo_path} 中搜索文件名"
        ))
        
        # search_code - 限定文件范围
        def search_code_in_subrepo(query: str) -> Dict[str, Any]:
            """在子仓库的代码中搜索文本"""
            subrepo_files = [f for f in repo.all_files if f.startswith(subrepo_path)]
            
            if not subrepo_files:
                return {"status": "error", "error_type": "NotFound", "message": "子仓库中没有文件"}
            
            return repo.search_code(query, files_to_search=subrepo_files)
        
        tools.append(FunctionTool(
            func=search_code_in_subrepo,
            name="search_code",
            description=f"在 {subrepo_path} 的代码中搜索文本"
        ))
        
        logger.info(f"为子仓库 '{subrepo_path}' 创建了 {len(tools)} 个受限工具")
        return tools


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 创建仓库实例
    linux_kernel_repo = Repo(
        local_path='/usr1/Rtos_Coding/data/repo/hm-verif-kernel',
        repo_name='/hm-verif-kernel',
     #   codehub_link='https://github.com/torvalds/linux'
    )
    
    # 创建工作空间
    workspace = RepoWorkSpace([linux_kernel_repo])
    
    # 获取全局工具
    global_tools = workspace.get_workspace_tools()
    print(f"全局工具数量: {len(global_tools)}")


    
    # 获取子仓库工具
    net_driver_tools = workspace.get_subrepo_tools("linux_kernel/drivers/net")
    print(f"子仓库工具数量: {len(net_driver_tools)}")
    
    for tool in net_driver_tools:
        print(f"  - {tool.name}: {tool.description}")
