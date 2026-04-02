import logging
from logging.handlers import TimedRotatingFileHandler, QueueHandler, QueueListener
from rich.logging import RichHandler
from queue import Queue
import os
import gzip
from rich.console import Console
from io import StringIO
import sys
import fcntl

LOG_FORMATTER = "%(message)s"
TIME_FORMATTER = "[%Y-%m-%d %H:%M:%S]"


class CompressedTimedRotatingFileHandler(TimedRotatingFileHandler):
    def __init__(self, filename, when='midnight', interval=1, backupCount=7, encoding='utf-8', delay=False, utc=False):
        super().__init__(
            filename=filename,
            when=when,
            interval=interval,
            backupCount=backupCount,
            encoding=encoding,
            delay=delay,
            utc=utc,
        )

    def doRollover(self):
        super().doRollover()
        self._compress_latest_backup()
        self._clean_old_compressed()

    def _compress_latest_backup(self):
        dir_name, base_name = os.path.split(self.baseFilename)
        backup_files = []
        for f in os.listdir(dir_name):
            if f.startswith(base_name + ".") and not f.endswith(".gz"):
                backup_files.append(os.path.join(dir_name, f))

        if backup_files:
            backup_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            latest_backup = backup_files[0]
            self._compress_file(latest_backup)
            os.remove(latest_backup)

    def _compress_file(self, source):
        try:
            with open(source, "rb") as f_in:
                compressed_name = source + ".gz"
                with gzip.open(compressed_name, "wb") as f_out:
                    f_out.writelines(f_in)
        except Exception as e:
            print(f"压缩失败: {e}")

    def _clean_old_compressed(self):
        dir_name, base_name = os.path.split(self.baseFilename)
        compressed_files = [
            os.path.join(dir_name, f)
            for f in os.listdir(dir_name)
            if f.startswith(base_name + ".") and f.endswith(".gz")
        ]
        compressed_files.sort(key=lambda x: os.path.getmtime(x))
        if len(compressed_files) > self.backupCount:
            for old_file in compressed_files[: -self.backupCount]:
                try:
                    os.remove(old_file)
                except Exception as e:
                    print(f"删除旧文件失败: {e}")


# _console = Console(file=StringIO(), force_terminal=False)  # 如果不需要终端控制符，可以关闭


def table_to_str(table):
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False)
    console.print(table)
    return buffer.getvalue()


def is_nonblocking(fd):
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    return bool(flags & os.O_NONBLOCK)


def setup_root_logging(
    logger_level=logging.INFO,
    default_log_path: str = "./applog",
    default_log_name: str = "app.log",
    console_output: bool = True,
    file_output: bool = True,
    third_party_lib_silence: bool = True,
):
    # 创建日志格式化器（仅用于文件日志）
    safe_console = Console(file=sys.stderr, force_terminal=True)
    formatter = logging.Formatter(LOG_FORMATTER, datefmt=TIME_FORMATTER)
    log_queue = Queue()
    queue_handler = QueueHandler(log_queue)
    # 初始化处理器列表
    handlers = []
    # 控制台处理器（如果启用）
    if console_output:
        # console_handler = logging.StreamHandler(sys.stderr)  # 普通终端输出流
        console_handler = RichHandler(console=safe_console, rich_tracebacks=True)
        console_handler.setLevel(logger_level)
        handlers.append(console_handler)
    # 文件处理器（如果启用）
    if file_output:
        file_handler = CompressedTimedRotatingFileHandler(
            filename=os.path.join(default_log_path, default_log_name), backupCount=7, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logger_level)
        handlers.append(file_handler)
    # 移除 root logger 所有现有 Handler
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    root_logger.setLevel(logger_level)
    root_logger.addHandler(queue_handler)
    # 如果没有启用任何输出处理器，添加一个NullHandler防止警告
    if not handlers:
        handlers.append(logging.NullHandler())
    # 创建队列监听器
    queue_listener = QueueListener(log_queue, *handlers, respect_handler_level=True)  # 解包处理器列表
    queue_listener.start()
    # 第三方库静默
    if third_party_lib_silence:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("metasearch-mcp").setLevel(logging.WARNING)

    print("stdout 非阻塞？", is_nonblocking(sys.stdout.fileno()))
    print("stderr 非阻塞？", is_nonblocking(sys.stderr.fileno()))


# 应用启动时调用
# setup_root_logging()
