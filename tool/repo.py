"""
代码仓库工作空间管理模块

本模块提供了统一的代码仓库管理接口，支持多仓库工作空间操作。
主要功能包括：
- 多仓库统一管理
- 基于路径的自动路由
- 工具函数的权限控制
- Git操作的安全管理
- 统一的工具接口导出

可以在此基础上，继承repo为CRepo, JavaRepo等，并创建属于自己的特有的tool



核心类：
- Repo: 单个代码仓库的封装，提供基础文件操作
- RepoWorkSpace: 多仓库工作空间，提供统一的工具接口和路由功能
"""

import os
import re
import pickle
import fnmatch
from pathlib import Path
from pydantic import BaseModel
from typing import Dict, List, Optional, Any, Literal
from thefuzz import process as fuzzy_process
from dataclasses import dataclass, field
import json
import logging

from repo_agent.core.types import Tool, FunctionTool, ToolExecutionResult, ToolCall

logger = logging.getLogger(__name__)



from repo_agent.core.types import FunctionTool

# Repo默认提供的工具列表，类似Linux命令
Repo_Default_Tool_List = [
    "search_files",        # 搜索文件名
    "search_code",         # 搜索代码内容
    "list_dir",           # 列出目录结构
    "read_file",          # 读取文件内容
 #   "create_file",        # 创建新文件
 #   "search_replace",     # 搜索并替换内容
]

Repo_Write_Tool_list = [
    "create_file",        # 创建新文件
    "search_replace",     # 搜索并替换内容

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
    表示一个代码仓库对象
    
    Repo内置的所有工具的路径都是repo内的完整相对路径，格式为：{repo_name}/path/to/file
    提供文件读写、搜索、目录浏览等基础操作，所有操作都基于仓库的相对路径。
    
    Attributes:
        local_path: 仓库在本地文件系统的绝对路径
        repo_name: 仓库名称，默认为目录名
        codehub_link: 远程代码仓库链接地址（如GitHub/GitLab URL）
        repo_expose_funcs: 该仓库允许的操作列表，默认为所有读操作
        approve_git_command: 是否允许执行Git命令
        approve_git_command_types: 允许的Git命令类型列表
        repo_cache_dir: 缓存目录路径
        repo_tool_execute_dir: 工具执行的中间产物存放目录
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
        """
        初始化一个Repo实例
        
        Args:
            local_path: 仓库本地路径
            repo_name: 仓库名称，为None时使用目录名
            codehub_link: 远程仓库链接
            repo_ignore_patterns: 忽略的文件模式列表
            scan_patterns: 扫描文件的模式列表
            repo_expose_funcs: 允许的操作列表，为None时使用默认列表
            approve_git_command: 是否允许Git操作
            approve_git_command_types: 允许的Git命令类型
            cache_dir: 缓存目录
            repo_description: 仓库描述
        """
        # 基础路径设置
        self.repo_dir = Path(local_path).resolve()
        if not self.repo_dir.is_dir():
            raise FileNotFoundError(f"仓库目录不存在: {self.repo_dir}")
        
        self.repo_name = repo_name or self.repo_dir.name

        # repo的本地路径
        self.local_path = str(self.repo_dir)
        self.codehub_link = codehub_link
        self._description = repo_description
        
        # 扫描和忽略模式
        self.repo_ignore_patterns = repo_ignore_patterns or []
        self.scan_patterns = scan_patterns
        
        # 可以允许的repo的工具函数
        self.repo_expose_funcs = repo_expose_funcs or Repo_Default_Tool_List.copy()
        self.approve_git_command = approve_git_command
        self.approve_git_command_types = approve_git_command_types or Safe_Git_Commands.copy()
        
        # 缓存和执行目录设置
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / '.repo_analyzer_cache'
        

        # repo tool的部分中间产物的存放地址
        self.repo_cache_dir = self.cache_dir / self.repo_name
        self.repo_tool_execute_dir = self.repo_cache_dir / 'execute'
        self.scan_cache_path = self.repo_cache_dir / 'scan.cache'
        
        # 创建必要的目录
        self.repo_cache_dir.mkdir(parents=True, exist_ok=True)
        self.repo_tool_execute_dir.mkdir(parents=True, exist_ok=True)
        
        # 内部状态
        self.all_files: List[str] = []
        self._is_scanned = False
        
        # 确保文件已扫描
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
    
    def _check_permission(self, func_name: str) -> bool:
        """检查是否有权限执行指定操作"""
        return func_name in self.repo_expose_funcs
    
    # ==================== 公共API方法 ====================
    
    def read_file(self, file_path: str) -> Dict[str, Any]:
        """
        读取仓库中指定文件的内容
        
        Args:
            file_path: 文件的相对路径，格式为 {repo_name}/path/to/file
        
        Returns:
            包含状态和数据的字典
        """
        if not self._check_permission("read_file"):
            return {"status": "error", "error_type": "PermissionError", "message": "当前仓库不允许读取文件操作"}
        
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
            包含目录树结构的字典
        """
        if not self._check_permission("list_dir"):
            return {"status": "error", "error_type": "PermissionError", "message": "当前仓库不允许列出目录操作"}
        
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
            包含搜索结果的字典
        """
        if not self._check_permission("search_files"):
            return {"status": "error", "error_type": "PermissionError", "message": "当前仓库不允许搜索文件操作"}
        
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
            包含搜索结果的字典
        """
        if not self._check_permission("search_code"):
            return {"status": "error", "error_type": "PermissionError", "message": "当前仓库不允许搜索代码操作"}
        
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
            操作结果字典
        """
        if not self._check_permission("create_file"):
            return {"status": "error", "error_type": "PermissionError", "message": "当前仓库不允许创建文件操作"}
        
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
            操作结果字典
        """
        if not self._check_permission("search_replace"):
            return {"status": "error", "error_type": "PermissionError", "message": "当前仓库不允许修改文件操作"}
        
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

