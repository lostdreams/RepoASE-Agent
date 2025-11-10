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
"""
"""
代码仓库工作空间管理模块 - 更新版

主要更新：
1. 添加了 run_git_command 工具函数
2. 改进了权限管理系统，支持 repo_approve_tools
3. 优化了工具函数的权限检查逻辑
"""

import os
import re
import pickle
import fnmatch
import subprocess
from pathlib import Path
from pydantic import BaseModel
from typing import Dict, List, Optional, Any, Literal
from thefuzz import process as fuzzy_process
from dataclasses import dataclass, field
import json
import logging

from repo_agent.core.types import Tool, FunctionTool, ToolExecutionResult, ToolCall

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
    "git push"
]


class Repo:
    """
    表示一个代码仓库对象
    
    提供文件读写、搜索、目录浏览、Git操作等基础功能。
    所有路径格式为：{repo_name}/path/to/file
    """
    
    def __init__(
        self,
        local_path: str | Path,
        repo_name: Optional[str] = None,
        codehub_link: Optional[str] = None,
        repo_ignore_patterns: Optional[List[str]] = None,
        scan_patterns: List[str] = ['**/*'],
        repo_approve_tools: Optional[List[str]] = None,
        approve_git_command: bool = False,
        approve_git_command_types: Optional[List[str]] = None,
        cache_dir: Optional[str | Path] = None,
        repo_description: Optional[str] = None,
    ):
        """初始化Repo实例"""
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
        if repo_approve_tools is None:
            self.repo_approve_tools = Repo_Default_Tool_List.copy()
        else:
            self.repo_approve_tools = repo_approve_tools
        
        # Git命令相关权限
        self.approve_git_command = approve_git_command
        if approve_git_command and "run_git_command" not in self.repo_approve_tools:
            self.repo_approve_tools.append("run_git_command")
        
        self.approve_git_command_types = approve_git_command_types or Safe_Git_Commands.copy()
        
        # 缓存和执行目录设置
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
    
    def _check_permission(self, func_name: str) -> Optional[Dict[str, Any]]:
        """
        检查是否有权限执行指定操作
        
        Returns:
            None 表示有权限，Dict 表示无权限并返回错误信息
        """
        if func_name not in self.repo_approve_tools:
            return {
                "status": "error",
                "error_type": "PermissionError",
                "message": f"当前仓库 '{self.repo_name}' 不允许 '{func_name}' 操作"
            }
        return None
    
    def _validate_git_command(self, command: str) -> Optional[Dict[str, Any]]:
        """验证Git命令是否被允许执行"""
        command = command.strip()
        
        if not command.startswith('git '):
            return {
                "status": "error",
                "error_type": "InvalidCommand",
                "message": "命令必须以 'git ' 开头"
            }
        
        is_allowed = False
        for allowed_cmd in self.approve_git_command_types:
            if command.startswith(allowed_cmd):
                is_allowed = True
                break
        
        if not is_allowed:
            return {
                "status": "error",
                "error_type": "PermissionError",
                "message": f"不允许执行此Git命令。允许的命令: {', '.join(self.approve_git_command_types)}"
            }
        
        dangerous_patterns = [
            r'--force',
            r'-f\b',
            r'rm\s',
            r'reset\s+--hard',
            r'clean\s+-[dfx]',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, command):
                return {
                    "status": "error",
                    "error_type": "DangerousOperation",
                    "message": f"检测到危险操作: '{pattern}'，已被阻止"
                }
        
        return None
    
    # ==================== 公共API方法（带详细docstring） ====================
    
    def read_file(self, file_path: str) -> Dict[str, Any]:
        """
        读取 {repo} 仓库中指定文件的内容
        
        此工具可以读取代码仓库中的任何文本文件。文件路径应该是相对于仓库根目录的相对路径，
        格式为：{repo_name}/path/to/file.ext
        
        功能说明：
        - 支持读取所有文本格式文件（.py, .java, .cpp, .md, .txt等）
        - 自动处理编码问题（UTF-8，错误字符会被忽略）
        - 如果文件不存在，会提供模糊匹配建议
        
        Args:
            file_path: 文件的相对路径，格式为 {repo_name}/path/to/file
                      例如：linux_kernel/drivers/net/ethernet/intel/e1000/e1000_main.c
        
        Returns:
            成功时返回：{"status": "success", "data": "文件完整内容"}
            失败时返回：{"status": "error", "error_type": "错误类型", "message": "错误描述", "suggestions": ["建议的文件路径"]}
        
        Example:
            # 读取Python文件
            read_file("my_project/src/main.py")
            
            # 读取README
            read_file("my_project/README.md")
            
            # 读取配置文件
            read_file("my_project/config/settings.json")
        
        注意事项：
        - 路径必须包含仓库名前缀
        - 大文件可能需要较长时间
        - 二进制文件会显示为乱码
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
        以树状结构列出 {repo} 仓库的目录内容
        
        此工具以树状图的形式展示仓库的目录结构，帮助你快速了解代码的组织方式。
        可以指定显示深度来控制展示的详细程度。
        
        功能说明：
        - 以树状图形式展示目录结构
        - 自动排序：目录在前，文件在后
        - 忽略配置的忽略文件（如.git, __pycache__等）
        - 支持任意深度的递归展示
        
        Args:
            start_dir: 起始目录路径，格式为 {repo_name}/path 或使用 "." 表示仓库根目录
                      例如：linux_kernel/drivers 或 "."
            depth: 显示深度，控制递归层级（默认为2）
                   depth=1 只显示直接子项
                   depth=2 显示两层
                   depth=3 显示三层，以此类推
        
        Returns:
            成功时返回：{"status": "success", "data": "树状目录结构"}
            失败时返回：{"status": "error", "error_type": "错误类型", "message": "错误描述"}
        
        Example:
            # 查看仓库根目录（深度2）
            list_dir("my_project", depth=2)
            输出示例：
            'my_project' [depth=2]
            ├── src
            │   ├── main.py
            │   └── utils.py
            ├── tests
            │   └── test_main.py
            └── README.md
            
            # 查看特定子目录
            list_dir("my_project/src", depth=1)
            
            # 深度扫描
            list_dir("my_project", depth=4)
        
        注意事项：
        - 深度过大可能导致输出过长
        - 隐藏文件和目录会被忽略
        - 对于大型仓库，建议从子目录开始查看
        """
        if err := self._check_permission("list_dir"):
            return err
        
        if start_dir in ['.', '/', self.repo_name, f'{self.repo_name}/']:
            start_path = self.repo_dir
        else:
            target_dir = start_dir
            if not target_dir.startswith(f"{self.repo_name}/"):
                return {
                    "status": "error",
                    "error_type": "InvalidPathError",
                    "message": f"路径必须以 '{self.repo_name}/' 开头"
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
        在 {repo} 仓库中按文件名搜索文件
        
        此工具使用模糊匹配算法搜索文件名，即使你不记得完整的文件名也能找到目标文件。
        支持按文件扩展名过滤，适合快速定位特定类型的文件。
        
        功能说明：
        - 使用模糊匹配算法（Fuzzy Matching），容错能力强
        - 支持部分匹配：输入"main"可以匹配"main.py", "test_main.cpp"等
        - 可按文件扩展名过滤（.py, .java, .cpp等）
        - 自动排序：按相似度从高到低排列
        
        Args:
            query: 搜索关键词，可以是文件名的一部分
                   例如："main", "test", "utils", "config"
            extensions: 文件扩展名过滤列表（可选）
                       例如：[".py", ".md"] 或 ["py", "md"]
                       不指定则搜索所有类型的文件
            limit: 返回结果数量限制（默认5个）
                   建议值：3-10之间
        
        Returns:
            成功时返回：{"status": "success", "data": ["匹配的文件路径列表"]}
            失败时返回：{"status": "error", "error_type": "NotFound", "message": "未找到匹配文件"}
        
        Example:
            # 搜索包含"main"的文件
            search_files("main")
            返回：["my_project/src/main.py", "my_project/test/test_main.py"]
            
            # 只搜索Python文件
            search_files("utils", extensions=[".py"])
            
            # 搜索配置文件
            search_files("config", extensions=[".json", ".yaml", ".yml"])
            
            # 搜索README
            search_files("readme", extensions=[".md"], limit=3)
        
        使用技巧：
        - 关键词越具体，匹配越精确
        - 使用扩展名过滤可以提高搜索效率
        - 如果结果太多，可以增加关键词或限制扩展名
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
        在 {repo} 仓库的代码文件中搜索指定的文本内容
        
        此工具在代码文件中搜索特定的文本、函数名、类名、变量名等，帮助你快速定位
        代码位置。支持全仓库搜索或限定文件范围搜索。
        
        功能说明：
        - 全文搜索：在文件内容中查找匹配的文本
        - 词边界匹配：精确匹配完整单词，避免部分匹配（如搜索"test"不会匹配"latest"）
        - 返回行号和上下文：每个匹配都包含文件路径、行号和行内容
        - 支持限定文件范围：可以只在指定文件中搜索
        
        Args:
            query: 搜索的文本内容
                   例如："def main", "class User", "import os", "TODO"
            files_to_search: 限定搜索的文件列表（可选）
                            格式为：["repo_name/path/to/file1.py", "repo_name/path/to/file2.py"]
                            不指定则搜索整个仓库的所有文件
        
        Returns:
            成功时返回：{"status": "success", "data": [
                {
                    "file_path": "文件路径",
                    "line_number": 行号,
                    "line_content": "该行的内容"
                },
                ...
            ]}
            失败时返回：{"status": "error", "error_type": "NotFound", "message": "未找到匹配内容"}
        
        Example:
            # 搜索函数定义
            search_code("def main")
            返回：[
                {"file_path": "my_project/src/main.py", "line_number": 15, "line_content": "def main():"},
                {"file_path": "my_project/tests/test.py", "line_number": 8, "line_content": "def main_test():"}
            ]
            
            # 搜索类定义
            search_code("class UserService")
            
            # 在指定文件中搜索
            search_code("import pandas", files_to_search=["my_project/src/data_processor.py"])
            
            # 搜索TODO注释
            search_code("TODO")
            
            # 搜索特定变量
            search_code("DATABASE_URL")
        
        使用技巧：
        - 搜索函数时包含"def"或"function"关键字
        - 搜索类时包含"class"关键字
        - 使用驼峰命名或下划线命名提高精确度
        - 对于大型仓库，建议先用search_files定位文件，再用此工具搜索内容
        
        注意事项：
        - 搜索词会进行词边界匹配（\b...\b）
        - 搜索是大小写敏感的
        - 二进制文件会被自动跳过
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
        在 {repo} 仓库中创建新文件
        
        此工具用于在代码仓库中创建新的文件。如果目标目录不存在，会自动创建。
        适用于生成代码文件、配置文件、文档等场景。
        
        功能说明：
        - 自动创建父目录：如果路径中的目录不存在，会自动创建
        - 防止覆盖：如果文件已存在，会返回错误，保护现有文件
        - 支持任意文本内容：代码、配置、文档等
        - 自动更新仓库文件索引
        
        Args:
            file_path: 要创建的文件路径，格式为 {repo_name}/path/to/new_file.ext
                      例如："my_project/src/new_module.py"
            file_content: 文件内容（字符串）
                         默认为空字符串（创建空文件）
        
        Returns:
            成功时返回：{"status": "success", "message": "文件创建成功"}
            失败时返回：{"status": "error", "error_type": "错误类型", "message": "错误描述"}
        
        Example:
            # 创建Python文件
            create_file("my_project/src/utils.py", '''
def hello():
    print("Hello, World!")
''')
            
            # 创建配置文件
            create_file("my_project/config.json", '{"debug": true, "port": 8080}')
            
            # 创建README
            create_file("my_project/docs/README.md", '''
# Documentation
This is the project documentation.
''')
            
            # 创建空文件
            create_file("my_project/data/output.txt")
        
        使用技巧：
        - 文件路径必须包含仓库名前缀
        - 使用多行字符串（'''）方便编写多行代码
        - 创建前可以先用list_dir检查目录结构
        
        注意事项：
        - 不能覆盖已存在的文件（会返回错误）
        - 如需修改文件，请使用search_replace工具
        - 路径分隔符统一使用 /
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
        在 {repo} 仓库的指定文件中搜索并替换文本内容
        
        此工具用于修改文件内容，通过搜索特定文本并替换为新文本。这是修改代码、
        配置文件的主要方式。支持精确匹配和全局替换。
        
        功能说明：
        - 全局替换：替换文件中所有匹配的文本
        - 精确匹配：只替换完全匹配的文本
        - 安全检查：替换前会检查文件是否存在
        - 返回结果：告知是否进行了修改
        
        Args:
            search_text: 要搜索的文本（必须完全匹配）
                        例如："old_function_name", "TODO: fix this", "version = '1.0'"
            replace_text: 替换后的新文本
                         例如："new_function_name", "DONE: fixed", "version = '2.0'"
            file_path: 目标文件路径，格式为 {repo_name}/path/to/file.ext
        
        Returns:
            成功时返回：{"status": "success", "message": "替换成功/未找到", "data": {"modified": true/false}}
            失败时返回：{"status": "error", "error_type": "错误类型", "message": "错误描述"}
        
        Example:
            # 修改函数名
            search_replace(
                "def old_function():",
                "def new_function():",
                "my_project/src/main.py"
            )
            
            # 更新配置值
            search_replace(
                '"debug": false',
                '"debug": true',
                "my_project/config.json"
            )
            
            # 修改文档
            search_replace(
                "## Version 1.0",
                "## Version 2.0",
                "my_project/README.md"
            )
            
            # 修改导入语句
            search_replace(
                "from old_module import func",
                "from new_module import func",
                "my_project/src/utils.py"
            )
        
        使用技巧：
        - 搜索文本要尽可能具体，避免误替换
        - 可以包含多行文本（使用'''）
        - 先用read_file查看文件内容，确认搜索文本
        - 替换前可以用search_code检查搜索文本的位置
        
        注意事项：
        - 搜索文本必须完全匹配（包括空格、换行）
        - 如果文件中不存在搜索文本，不会报错但会返回modified=false
        - 替换是全局的，会替换所有匹配项
        - 无法撤销，请谨慎使用
        
        最佳实践：
        1. 先用read_file读取文件内容
        2. 确认要替换的精确文本
        3. 执行search_replace
        4. 再用read_file确认修改结果
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
    
    def run_git_command(self, command: str) -> Dict[str, Any]:
        """
        在 {repo} 仓库目录下执行Git命令  该命令会在该仓库的实际路径下进行执行并返回结果
        
        
        此工具允许执行安全的Git命令来管理代码仓库。支持查看状态、历史记录、
        差异对比、分支管理等操作。所有命令都经过严格的安全验证。
        
        功能说明：
        - 在仓库目录下执行Git命令
        - 多层安全保护：命令白名单、危险操作检测、超时保护
        - 返回详细结果：包括标准输出、错误输出、返回码
     
        Args:
            command: Git命令字符串（必须以"git "开头）
                    例如："git status", "git log --oneline -10", "git diff HEAD~1"

        Returns:
            成功时返回：{"status": "success", "data": {
                "stdout": "标准输出",
                "stderr": "错误输出",
                "returncode": 返回码,
                "command": "执行的命令"
            }}
            失败时返回：{"status": "error", "error_type": "错误类型", "message": "错误描述"}
        
        Example:
            # 查看仓库状态
            run_git_command("git status")
            返回当前工作区状态
            
            run_git_command("git log --oneline -5")
            返回最近5条提交记录
            
            run_git_command("git diff README.md")
            查看README.md的修改
            
            # 查看分支列表
            run_git_command("git branch -a")
            列出所有分支
            
            # 查看特定提交
            run_git_command("git show HEAD")
            查看最新提交的详情
            
            # 暂存文件
            run_git_command("git add src/main.py")
            
            # 提交更改
            run_git_command('git commit -m "Update main.py"')
        
        支持的安全命令：
        - git status: 查看仓库状态
        - git log: 查看提交历史
        - git diff: 查看差异
        - git show: 查看提交详情
        - git branch: 分支管理
        - git add: 暂存更改
        - git commit: 提交更改
        
        危险命令（会被阻止）：
        - 包含 --force 或 -f 的命令
        - git reset --hard
        - git clean -df
        - git rm -rf
        
        使用技巧：
        - 使用 -n 参数可以限制输出数量（如git log -n 10）
        - 使用 --oneline 参数可以简化输出
        - 组合使用多个参数获取精确信息
        
        注意事项：
        - 命令必须以"git "开头
        - 命令必须在白名单中
        - 超时时间为30秒
        - 危险操作会被自动拦截
        """
        if err := self._check_permission("run_git_command"):
            return err
        
        if err := self._validate_git_command(command):
            return err
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(self.repo_dir),
                capture_output=True,
                text=True,
                timeout=30,
                env=os.environ.copy()
            )
            
            return {
                "status": "success",
                "data": {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                    "command": command
                }
            }
        
        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "error_type": "TimeoutError",
                "message": f"命令执行超时（30秒）: {command}"
            }
        except Exception as e:
            return {
                "status": "error",
                "error_type": e.__class__.__name__,
                "message": f"执行Git命令失败: {e}"
            }
    
    # ==================== 工具导出方法 ====================
    
    def get_repo_tools(self) -> List[str]:
        """
        获取当前仓库支持的工具函数名称列表
        
        Returns:
            工具函数名称列表，这些函数名对应repo_approve_tools中的权限
        
        Example:
            >>> repo.get_repo_tools()
            ['read_file', 'list_dir', 'search_files', 'search_code', 'create_file']
        """
        # 工具名称映射
        tool_mapping = {
            "read_file": "read_file",
            "list_dir": "list_dir",
            "search_files": "search_files",
            "search_code": "search_code",
            "create_file": "create_file",
            "search_replace": "search_replace",
            "run_git_command": "run_git_command"
        }
        
        # 只返回有权限的工具
        available_tools = []
        for tool_name in self.repo_approve_tools:
            if tool_name in tool_mapping:
                available_tools.append(tool_mapping[tool_name])
        
        return available_tools
