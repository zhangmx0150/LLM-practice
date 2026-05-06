from unstructured.partition.pdf import partition_pdf
from collections import Counter

# PDF文件路径
pdf_path = "../../data/C2/pdf/rag.pdf"

# ==========================================
# 切换解析策略：修改此处的字符串即可
# 可选值: "hi_res" 或 "ocr_only"
# ==========================================
STRATEGY = "ocr_only"

print(f"⏳ 正在使用 {STRATEGY} 策略解析 PDF，这可能需要一点时间...\n")

# 使用 partition_pdf 加载并解析PDF文档
elements = partition_pdf(
    filename=pdf_path,
    # strategy=STRATEGY,
    # 强烈建议：如果是中文文档，必须指定 OCR 语言，否则 Tesseract 默认用英文识别会导致乱码
    languages=["chi_sim", "eng"]
)

# 打印解析结果
print(f"✅ 解析完成: 共提取 {len(elements)} 个元素, 总计 {sum(len(str(e)) for e in elements)} 字符\n")

# 统计元素类型
types = Counter(e.category for e in elements)
print(f"📊 元素类型统计: {dict(types)}\n")

# 显示所有元素 (如果元素太多，可以考虑切片，例如 elements[:10] 只看前 10 个)
print("📃 元素详情:")
for i, element in enumerate(elements, 1):
    print(f"Element {i} ({element.category}):")
    print(element)
    print("=" * 60)