"""
知识库
"""
import json
import os
import io
import re
import config_data as config
import hashlib
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
from typing import Optional

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None
try:
    import markdown as md_lib
except ImportError:
    md_lib = None


def check_md5(md5_str: str):
    """检查传入的MD5字符串是否已经被处理过了
    return False未处理，True已处理
    """
    if not os.path.exists(config.md5_path):
        open(config.md5_path, 'w', encoding='utf-8').close()
        return False
    else:
        for line in open(config.md5_path, 'r', encoding='utf-8').readlines():
            line = line.strip()
            if line == md5_str:
                return True
        return False


def save_md5(md5_str: str):
    """将传入的md5字符串，记录到文件内保存"""
    with open(config.md5_path, 'a', encoding="utf-8") as f:
        f.write(md5_str + '\n')


def get_string_md5(input_str: str, encoding='utf-8'):
    """将传入的字符串转换为md5字符串"""
    str_bytes = input_str.encode(encoding=encoding)
    md5_obj = hashlib.md5()
    md5_obj.update(str_bytes)
    md5_hex = md5_obj.hexdigest()
    return md5_hex


# =================== 文件解析器 ===================

SUPPORTED_EXTENSIONS = {'.txt', '.pdf', '.json', '.jsonl', '.md', '.markdown'}
SUPPORTED_DESCRIPTION = "txt, pdf, json, jsonl, md"


def parse_file(content_bytes: bytes, filename: str) -> str:
    """
    根据文件扩展名自动识别并解析文件内容为纯文本。
    content_bytes: 文件的原始字节内容
    filename: 文件名（用于判断扩展名）
    返回: 解析后的纯文本字符串
    """
    ext = os.path.splitext(filename)[1].lower()

    if ext == '.txt':
        return _parse_txt(content_bytes)
    elif ext == '.pdf':
        return _parse_pdf(content_bytes)
    elif ext == '.json':
        return _parse_json(content_bytes)
    elif ext == '.jsonl':
        return _parse_jsonl(content_bytes)
    elif ext in ('.md', '.markdown'):
        return _parse_markdown(content_bytes)
    else:
        raise ValueError(
            f"不支持的文件格式：{ext}，"
            f"支持格式：{', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )


def get_file_label(ext: str) -> str:
    """返回文件格式的中文标签"""
    labels = {
        '.txt': '纯文本',
        '.pdf': 'PDF 文档',
        '.json': 'JSON 数据',
        '.jsonl': 'JSONL 数据',
        '.md': 'Markdown 文档',
        '.markdown': 'Markdown 文档',
    }
    return labels.get(ext.lower(), ext)


def _decode_bytes(content_bytes: bytes) -> str:
    """尝试用 UTF-8 / GBK 解码字节"""
    for enc in ('utf-8', 'gbk', 'utf-16'):
        try:
            return content_bytes.decode(enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return content_bytes.decode('utf-8', errors='replace')


def _parse_txt(content_bytes: bytes) -> str:
    """解析 TXT 文件"""
    return _decode_bytes(content_bytes)


def _parse_pdf(content_bytes: bytes) -> str:
    """
    解析 PDF 文件，提取全部文本内容。
    依赖 pypdf 库，未安装时给出明确提示。
    """
    if PdfReader is None:
        raise ImportError("解析 PDF 需要 pypdf 库。请运行：pip install pypdf")
    pdf_file = io.BytesIO(content_bytes)
    reader = PdfReader(pdf_file)
    pages_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text)
    return "\n\n".join(pages_text)


def _extract_json_text(obj, indent: int = 0) -> str:
    """
    递归提取 JSON 数据中所有有意义的文本内容。
    处理嵌套字典、列表，将键值对格式化为可读文本。
    """
    prefix = "  " * indent
    parts = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, (dict, list)):
                nested = _extract_json_text(value, indent + 1)
                if nested.strip():
                    parts.append(f"{prefix}{key}:\n{nested}")
            elif isinstance(value, str) and value.strip():
                parts.append(f"{prefix}{key}: {value}")
            elif value is not None:
                parts.append(f"{prefix}{key}: {value}")
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                nested = _extract_json_text(item, indent + 1)
                if nested.strip():
                    parts.append(f"{prefix}- {nested.strip()}")
            elif isinstance(item, str) and item.strip():
                parts.append(f"{prefix}- {item}")
            elif item is not None:
                parts.append(f"{prefix}- {item}")
    elif isinstance(obj, str) and obj.strip():
        parts.append(f"{prefix}{obj}")

    return "\n".join(parts)


def _parse_json(content_bytes: bytes) -> str:
    """解析 JSON 文件"""
    text = _decode_bytes(content_bytes)
    data = json.loads(text)
    return _extract_json_text(data)


def _parse_jsonl(content_bytes: bytes) -> str:
    """解析 JSONL 文件（每行一个 JSON 对象）"""
    text = _decode_bytes(content_bytes)
    lines = []
    for i, line in enumerate(text.strip().splitlines()):
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        extracted = _extract_json_text(data)
        if extracted.strip():
            lines.append(f"[第 {i+1} 条记录]\n{extracted}")
    return "\n\n".join(lines)


def _parse_markdown(content_bytes: bytes) -> str:
    """解析 Markdown 文件，返回纯文本内容"""
    text = _decode_bytes(content_bytes)
    if md_lib is None:
        return text
    html = md_lib.markdown(text)
    plain = re.sub(r'<[^>]+>', '', html)
    plain = plain.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    plain = plain.replace('&quot;', '"').replace('&#39;', "'")
    plain = re.sub(r'\n{3,}', '\n\n', plain)
    return plain.strip()


class KnowledgeBaseService(object):
    def __init__(self):
        os.makedirs(config.persist_directory, exist_ok=True)

        self.chroma = Chroma(
            collection_name=config.collection_name,
            embedding_function=DashScopeEmbeddings(model="text-embedding-v4"),
            persist_directory=config.persist_directory,
        )
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
            length_function=len,
        )

    def upload_by_str(self, data: str, filename):
        """将传入的字符串，进行向量化，存入向量数据库中"""
        md5_hex = get_string_md5(data)
        if check_md5(md5_hex):
            return "[Repeat] 内容已存在知识库"
        if len(data) > config.max_spliter_char_number:
            knowledge_chunks: list[str] = self.spliter.split_text(data)
        else:
            knowledge_chunks = [data]

        metadata = {
            "source": filename,
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator": "客户",
        }

        self.chroma.add_texts(
            knowledge_chunks,
            metadata=[metadata for _ in knowledge_chunks],
        )
        save_md5(md5_hex)
        return "[Success] 内容已经成功载入向量库"

    def upload_by_file(self, content_bytes: bytes, filename: str):
        """
        解析文件内容并载入知识库。
        支持 TXT / PDF / JSON / JSONL / Markdown 格式。
        """
        text = parse_file(content_bytes, filename)
        return self.upload_by_str(text, filename)


if __name__ == '__main__':
    service = KnowledgeBaseService()
    r = service.upload_by_str("流星", "testfile")
    print(r)
