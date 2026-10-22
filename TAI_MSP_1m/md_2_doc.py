# --*-- conding:utf-8 --*--
# @File  : md_2_doc.py
# @Author: FH
# @Date  : 2026/10/13
# @Desc  :
import pypandoc

input_md = "file.md"
output_docx = "辐射定标技术文档.docx"

pypandoc.convert_file(
    input_md,
    "docx",
    outputfile=output_docx,
    extra_args=[
        f"--reference-doc={'D:/SD_WORK/2m_PSI/Test Specifications and Procedures-模板/fh/3_4_Test Specification_v2.0_unpacker.docx'}",
        "--toc"
    ]
)

print("已生成：", output_docx)