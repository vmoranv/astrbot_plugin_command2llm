from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
import asyncio
import json
import re
from difflib import SequenceMatcher
import uuid
import inspect

# 重新导入agent相关模块
try:
    from astrbot.core.agent.run_context import ContextWrapper
    from astrbot.core.agent.tool import FunctionTool, ToolExecResult, ToolSet
    from astrbot.core.astr_agent_context import AstrAgentContext
    from pydantic import Field
    from pydantic.dataclasses import dataclass
    AGENT_AVAILABLE = True
except ImportError:
    logger.warning("Agent模块不可用，使用备用实现")
    AGENT_AVAILABLE = False



# 尝试导入star相关模块
try:
    from astrbot.core.star.star_handler import star_handlers_registry
    from astrbot.core.star.star import star_map
    from astrbot.core.star.filter.command import CommandFilter
    from astrbot.core.star.filter.command_group import CommandGroupFilter
    from astrbot.core.star.filter.regex import RegexFilter
    from astrbot.core.star.filter.permission import PermissionTypeFilter
    STAR_AVAILABLE = True
except ImportError:
    logger.warning("Star模块不可用，使用备用实现")
    STAR_AVAILABLE = False
    star_handlers_registry = []
    star_map = {}
    CommandFilter = None
    CommandGroupFilter = None
    RegexFilter = None
    PermissionTypeFilter = None

# 尝试导入EventType
try:
    from astrbot.core.star_handler import EventType
except ImportError:
    try:
        from astrbot.core.star import EventType
    except ImportError:
        # 如果都失败了，定义一个临时的EventType
        class EventType:
            AdapterMessageEvent = "AdapterMessageEvent"

@register("command2llm", "vmoranv", "让大模型能够调用所有插件命令的插件", "1.0.0")
class Command2LLMPlugin(Star):
    def __init__(self, context: Context):
        super().__init__(context)
        self.command_cache = {}
        self.cache_timeout = 300  # 缓存5分钟
        self.enabled = True  # 插件开关
        self.threshold = 0.6  # 命令匹配阈值
        self.last_messages = {}  # 存储用户最后的消息
        self.processed_message_ids = set()  # 存储已处理的消息ID，避免重复处理
        

    async def initialize(self):
        """插件初始化方法"""
        if not STAR_AVAILABLE:
            logger.warning("Star模块不可用，插件功能受限")
        logger.info("Command2LLM插件初始化完成")

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def handle_message(self, event: AstrMessageEvent):
        """拦截所有消息，判断是否需要调用命令"""
        try:
            # 检查插件是否启用
            if not self.enabled:
                return

            message_str = event.message_str.strip()
            session_id = event.session_id
            
            
            
            # 首先检查是否是我们伪造的事件（通过session_id标记）
            if hasattr(event, 'session_id') and event.session_id.endswith("_cmd2llm"):
                logger.info(f"跳过自己伪造的事件: {event.session_id}")
                return
            
            # 获取消息ID进行检查
            message_id = None
            if hasattr(event, 'message_id'):
                message_id = event.message_id
            elif hasattr(event, 'message_obj') and hasattr(event.message_obj, 'message_id'):
                message_id = event.message_obj.message_id
            
            # 检查是否已经处理过这个消息（避免处理自己伪造的事件）
            if message_id and message_id in self.processed_message_ids:
                logger.info(f"跳过已处理的消息ID: {message_id}")
                return

            # 标记此消息ID为已处理
            if message_id:
                self.processed_message_ids.add(message_id)
                logger.info(f"标记消息ID为已处理: {message_id}")
                # 清理过期的消息ID（保持集合大小合理）
                if len(self.processed_message_ids) > 100:
                    self.processed_message_ids.clear()

            # 跳过命令消息（避免与现有命令冲突）
            if message_str.startswith('/') or message_str.startswith('#') or message_str.startswith('!'):
                # 存储最后一条消息（用于命令纠正）
                if not message_str.startswith('!'):  # 不存储本插件的命令
                    self.last_messages[session_id] = message_str
                return

            # 跳过本插件的控制命令
            control_commands = ['ai_enable', 'ai_disable', 'ai_status', 'refresh_commands']
            if any(message_str == f'!{cmd}' or message_str == cmd for cmd in control_commands):
                return

            # 存储最后一条消息
            self.last_messages[session_id] = message_str

            # 获取当前会话使用的聊天模型ID
            try:
                umo = event.unified_msg_origin
                provider_id = await self.context.get_current_chat_provider_id(umo=umo)
            except:
                return  # 无法获取提供商时跳过
            
            if not provider_id:
                return  # 没有LLM提供商时跳过

            # 检查消息是否需要调用命令
            should_call = await self._should_call_command(event, provider_id)
            logger.info(f"是否需要调用命令: {should_call}")
            if not should_call:
                return

            # 检查agent模块是否可用
            if not AGENT_AVAILABLE:
                logger.error("Agent模块不可用")
                yield event.plain_result("抱歉，Agent模块不可用")
                return

            # 获取所有可用命令列表
            commands_info = self._get_all_commands_info()
            commands_list = []
            for plugin_name, cmd_list in commands_info.items():
                for cmd in cmd_list:
                    command_name = cmd.split('#')[0].strip()
                    if command_name:
                        commands_list.append(command_name)
            
            # 添加调试日志
            logger.info(f"可用命令列表: {commands_list[:10]}...")  # 只显示前10个

            # 构建系统提示，包含可用命令列表
            system_prompt = f"""你是一个命令执行助手，只负责调用插件命令。

可用命令列表：
{chr(10).join(commands_list[:20])}  # 只显示前20个命令避免过长

当用户需要执行某个命令时，请使用execute_command工具来执行命令。
例如：用户说"帮我分析群聊"，你应该调用execute_command工具，参数为"群分析 7"
如果用户问"今天吃什么"，而命令列表中有"今天吃什么"，你应该调用execute_command工具，参数为"今天吃什么"

重要规则：
1. 使用工具执行命令时，参数只需要命令名和参数，不需要加/前缀
2. 确保命令在可用列表中
3. 执行命令后不要回复任何内容，不要解释，不要建议，不要聊天
4. 只执行命令，不进行任何对话"""

            # 创建工具集合
            tools = ToolSet([
                ExecuteCommandTool(self.context)
            ])

            # 调用Agent处理消息
            try:
                llm_resp = await self.context.tool_loop_agent(
                    event=event,
                    chat_provider_id=provider_id,
                    prompt=message_str,
                    system_prompt=system_prompt,
                    tools=tools,
                    max_steps=5,
                    tool_call_timeout=60
                )

                # 发送AI回复
                logger.info(f"Agent回复: {llm_resp.completion_text}")
                yield event.plain_result(llm_resp.completion_text)
            except Exception as e:
                logger.error(f"Agent调用失败: {str(e)}")
                yield event.plain_result("抱歉，我暂时无法处理您的请求。请稍后再试。")
                
        except Exception as e:
            logger.error(f"消息处理错误: {str(e)}")

    async def _should_call_command(self, event: AstrMessageEvent, provider_id: str) -> bool:
        """判断是否需要调用命令"""
        try:
            message_str = event.message_str.strip()
            
            # 简单的启发式判断
            call_keywords = [
                '帮我', '请', '能否', '可以', '能不能', '如何', '怎么', '怎样',
                '查看', '搜索', '找', '获取', '设置', '配置', '启动', '停止',
                '天气', '时间', '日期', '新闻', '音乐', '视频', '图片'
            ]
            
            # 如果消息包含调用关键词，则返回True
            for keyword in call_keywords:
                if keyword in message_str:
                    logger.info(f"匹配到关键词: {keyword}")
                    return True
            
            # 使用LLM进行更精确的判断
            try:
                llm_resp = await self.context.llm_generate(
                    chat_provider_id=provider_id,
                    prompt=f"请判断以下消息是否需要调用某个命令或工具来处理：'{message_str}'\n只需要回答'是'或'否'。",
                    system_prompt="你是一个消息分类器，判断用户消息是否需要调用命令或工具。"
                )
                
                result = llm_resp.completion_text.strip()
                logger.info(f"LLM判断结果: {result}")
                return '是' in result
            except:
                return False
            
        except Exception as e:
            logger.error(f"判断命令调用时出错: {str(e)}")
            return False

    @filter.command("ai_enable", alias={"!ai_enable"})
    async def ai_enable(self, event: AstrMessageEvent):
        """启用AI自动调用命令功能"""
        self.enabled = True
        yield event.plain_result("AI自动调用命令功能已启用")

    @filter.command("ai_disable", alias={"!ai_disable"})
    async def ai_disable(self, event: AstrMessageEvent):
        """禁用AI自动调用命令功能"""
        self.enabled = False
        yield event.plain_result("AI自动调用命令功能已禁用")

    @filter.command("ai_status", alias={"!ai_status"})
    async def ai_status(self, event: AstrMessageEvent):
        """查看AI功能状态"""
        status = "启用" if self.enabled else "禁用"
        star_status = "可用" if STAR_AVAILABLE else "不可用"
        yield event.plain_result(f"AI自动调用命令功能当前状态: {status}\nStar模块: {star_status}")

    @filter.command("refresh_commands", alias={"!refresh_commands"})
    async def refresh_commands(self, event: AstrMessageEvent):
        """刷新命令缓存"""
        self.command_cache.clear()
        yield event.plain_result("命令缓存已刷新")

    

    def _get_all_commands_info(self) -> dict:
        """获取所有其他插件及其命令列表, 参考help插件的实现"""
        import collections
        plugin_commands = collections.defaultdict(list)
        
        try:
            # 获取所有插件的元数据，并且去掉未激活的
            all_stars_metadata = self.context.get_all_stars()
            all_stars_metadata = [star for star in all_stars_metadata if star.activated]
        except Exception as e:
            logger.error(f"获取插件列表失败: {e}")
            return {}
        
        if not all_stars_metadata:
            logger.warning("没有找到任何插件")
            return {}
        
        for star in all_stars_metadata:
            plugin_name = getattr(star, "name", "未知插件")
            plugin_instance = getattr(star, "star_cls", None)
            module_path = getattr(star, "module_path", None)
            
            # 跳过自身和核心插件
            if (plugin_name == "astrbot" or 
                plugin_name == "astrbot_plugin_command2llm" or 
                plugin_name == "astrbot-reminder"):
                continue
            
            # 进行必要的检查
            if (not plugin_name or not module_path or 
                not isinstance(plugin_instance, Star)):
                logger.warning(f"插件 '{plugin_name}' (模块: {module_path}) 的元数据无效或不完整，已跳过。")
                continue
            
            # 检查插件实例是否是当前插件的实例 (排除自身)
            if plugin_instance is self:
                continue
            
            # 遍历所有注册的处理器
            for handler in star_handlers_registry:
                # 确保处理器元数据有效且类型正确
                if not hasattr(handler, 'handler_module_path'):
                    continue
                
                # 检查此处理器是否属于当前遍历的插件 (通过模块路径匹配)
                if handler.handler_module_path != module_path:
                    continue
                
                command_name = None
                description = getattr(handler, 'desc', None)
                
                # 遍历处理器的过滤器，查找命令或命令组
                if hasattr(handler, 'event_filters'):
                    for filter_ in handler.event_filters:
                        if CommandFilter and isinstance(filter_, CommandFilter):
                            # 获取父命令名
                            parent_names = getattr(filter_, 'parent_command_names', [''])
                            full_command = f"{parent_names[0]} {filter_.command_name}".strip()
                            command_name = full_command
                            break
                        elif CommandGroupFilter and isinstance(filter_, CommandGroupFilter):
                            command_name = filter_.group_name
                            break
                
                # 如果找到了命令或命令组名称
                if command_name:
                    # 格式化字符串
                    if description:
                        formatted_command = f"{command_name}#{description}"
                    else:
                        formatted_command = command_name
                    
                    # 将格式化后的命令添加到对应插件的列表中
                    if formatted_command not in plugin_commands[plugin_name]:
                        plugin_commands[plugin_name].append(formatted_command)
        
        return dict(plugin_commands)

    def _get_all_available_commands(self) -> list:
        """获取所有可用命令列表"""
        try:
            commands_info = self._get_all_commands_info()
            commands = []
            for plugin_name, cmd_list in commands_info.items():
                for cmd in cmd_list:
                    # 提取命令名（去掉描述部分）
                    command_name = cmd.split('#')[0].strip()
                    if command_name:
                        commands.append(command_name)
            return commands
        except Exception as e:
            logger.error(f"获取命令列表失败: {str(e)}")
            return []

    def _find_best_command_match(self, message: str, commands: list) -> tuple:
        """查找最佳匹配的命令"""
        best_match = None
        best_ratio = 0
        
        # 提取消息的第一部分作为命令候选
        message_parts = message.split()
        if not message_parts:
            return None
        
        command_candidate = message_parts[0]
        
        for cmd in commands:
            ratio = SequenceMatcher(None, command_candidate, cmd).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = (cmd, ratio)
        
        return best_match

    async def terminate(self):
        """插件销毁方法"""
        logger.info("Command2LLM插件已卸载")


class ExecuteCommandTool(FunctionTool[AstrAgentContext]):
    """执行插件命令的工具"""
    
    def __init__(self, context: Context):
        self.context = context
        self.name = "execute_command"
        self.description = "执行指定的插件命令"
        self.parameters = {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "要执行的命令，包括命令名称和参数，例如：'今天吃什么' 或 '群分析 7'"
                }
            },
            "required": ["command"]
        }

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> ToolExecResult:
        try:
            command = kwargs.get("command", "").strip()
            if not command:
                return "命令不能为空"
            
            # 获取事件对象
            event = context.context.event
            
            # 通过框架执行命令
            try:
                from astrbot.core.platform import platform
                platform_name = event.get_platform_name()
                platform_obj = self.context.get_platform(platform_name)
                
                if not platform_obj:
                    logger.error("无法获取平台对象")
                    return "内部错误，无法确定平台"
                
                # 构造伪造的 AstrBotMessage
                from astrbot.api.platform import AstrBotMessage
                from astrbot.api.message_components import Plain
                
                fake_message = AstrBotMessage()
                fake_message.type = event.message_obj.type if hasattr(event, 'message_obj') and event.message_obj else "text"
                fake_message.message_str = f"/{command}"
                fake_message.message = [Plain(text=f"/{command}")]
                fake_message.self_id = event.message_obj.self_id if hasattr(event, 'message_obj') and event.message_obj else 0
                fake_message.session_id = event.session_id + "_cmd2llm"
                fake_message.message_id = str(uuid.uuid4())
                
                # 设置发送者信息
                if hasattr(event, 'message_obj') and event.message_obj and hasattr(event.message_obj, 'sender'):
                    fake_message.sender = event.message_obj.sender
                
                # 创建伪造事件
                OriginalEventClass = event.__class__
                
                # 检查构造函数签名并准备参数
                kwargs_event = {
                    "message_str": f"/{command}",
                    "message_obj": fake_message,
                    "platform_meta": platform_obj.meta(),
                    "session_id": event.session_id,
                }
                
                try:
                    sig = inspect.signature(OriginalEventClass.__init__)
                    if 'bot' in sig.parameters:
                        kwargs_event['bot'] = event.bot
                except Exception as e:
                    logger.warning(f"无法检查事件构造函数签名: {e}")
                
                fake_event = OriginalEventClass(**kwargs_event)
                
                logger.info(f"通过工具执行命令: /{command}")
                platform_obj.commit_event(fake_event)
                logger.info(f"命令 /{command} 已提交到框架执行")
                
                return f"命令 '{command}' 已执行"
                
            except Exception as e:
                logger.error(f"执行命令时出错: {str(e)}")
                return f"执行命令失败: {str(e)}"
                
        except Exception as e:
            logger.error(f"工具调用失败: {str(e)}")
            return f"工具调用失败: {str(e)}"




