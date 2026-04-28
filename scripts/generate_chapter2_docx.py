from __future__ import annotations

import argparse
import datetime as dt
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape


Paragraph = tuple[str, str]


def build_paragraph(text: str, style: str = "Normal", align: str | None = None) -> str:
    text_xml = escape(text)
    p_pr = [f'<w:pStyle w:val="{style}"/>']
    if align:
        p_pr.append(f'<w:jc w:val="{align}"/>')
    return (
        "<w:p>"
        f"<w:pPr>{''.join(p_pr)}</w:pPr>"
        f"<w:r><w:t xml:space=\"preserve\">{text_xml}</w:t></w:r>"
        "</w:p>"
    )


def build_document_xml(paragraphs: list[Paragraph]) -> str:
    body = []
    for style, text in paragraphs:
        align = "center" if style in {"Title", "Formula"} else None
        body.append(build_paragraph(text, style=style, align=align))
    body.append(
        "<w:sectPr>"
        "<w:pgSz w:w=\"11906\" w:h=\"16838\"/>"
        "<w:pgMar w:top=\"1440\" w:right=\"1800\" w:bottom=\"1440\" w:left=\"1800\" "
        "w:header=\"720\" w:footer=\"720\" w:gutter=\"0\"/>"
        "</w:sectPr>"
    )
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<w:document "
        "xmlns:wpc=\"http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas\" "
        "xmlns:mc=\"http://schemas.openxmlformats.org/markup-compatibility/2006\" "
        "xmlns:o=\"urn:schemas-microsoft-com:office:office\" "
        "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\" "
        "xmlns:m=\"http://schemas.openxmlformats.org/officeDocument/2006/math\" "
        "xmlns:v=\"urn:schemas-microsoft-com:vml\" "
        "xmlns:wp14=\"http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing\" "
        "xmlns:wp=\"http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing\" "
        "xmlns:w10=\"urn:schemas-microsoft-com:office:word\" "
        "xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\" "
        "xmlns:w14=\"http://schemas.microsoft.com/office/word/2010/wordml\" "
        "xmlns:wpg=\"http://schemas.microsoft.com/office/word/2010/wordprocessingGroup\" "
        "xmlns:wpi=\"http://schemas.microsoft.com/office/word/2010/wordprocessingInk\" "
        "xmlns:wne=\"http://schemas.microsoft.com/office/word/2006/wordml\" "
        "xmlns:wps=\"http://schemas.microsoft.com/office/word/2010/wordprocessingShape\" "
        "mc:Ignorable=\"w14 wp14\">"
        f"<w:body>{''.join(body)}</w:body>"
        "</w:document>"
    )


def build_styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:docDefaults>
    <w:rPrDefault>
      <w:rPr>
        <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="宋体"/>
        <w:sz w:val="24"/>
        <w:szCs w:val="24"/>
      </w:rPr>
    </w:rPrDefault>
    <w:pPrDefault>
      <w:pPr>
        <w:spacing w:line="420" w:lineRule="auto" w:before="0" w:after="120"/>
      </w:pPr>
    </w:pPrDefault>
  </w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:ind w:firstLine="420"/>
      <w:spacing w:line="420" w:lineRule="auto" w:before="0" w:after="120"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="宋体"/>
      <w:sz w:val="24"/>
      <w:szCs w:val="24"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:jc w:val="center"/>
      <w:spacing w:before="0" w:after="240"/>
      <w:ind w:firstLine="0"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="黑体"/>
      <w:b/>
      <w:sz w:val="32"/>
      <w:szCs w:val="32"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:spacing w:before="240" w:after="120"/>
      <w:ind w:firstLine="0"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="黑体"/>
      <w:b/>
      <w:sz w:val="28"/>
      <w:szCs w:val="28"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:spacing w:before="180" w:after="120"/>
      <w:ind w:firstLine="0"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="黑体"/>
      <w:b/>
      <w:sz w:val="26"/>
      <w:szCs w:val="26"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Formula">
    <w:name w:val="Formula"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:jc w:val="center"/>
      <w:spacing w:before="60" w:after="60"/>
      <w:ind w:firstLine="0"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math" w:eastAsia="宋体"/>
      <w:i/>
      <w:sz w:val="24"/>
      <w:szCs w:val="24"/>
    </w:rPr>
  </w:style>
</w:styles>
"""


def build_content_types_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
"""


def build_root_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""


def build_document_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>
"""


def build_core_xml(now: dt.datetime) -> str:
    created = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
  xmlns:dc="http://purl.org/dc/elements/1.1/"
  xmlns:dcterms="http://purl.org/dc/terms/"
  xmlns:dcmitype="http://purl.org/dc/dcmitype/"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>第二章 方法框架</dc:title>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{created}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{created}</dcterms:modified>
</cp:coreProperties>
"""


def build_app_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
  xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Microsoft Office Word</Application>
  <DocSecurity>0</DocSecurity>
  <ScaleCrop>false</ScaleCrop>
  <Company></Company>
  <LinksUpToDate>false</LinksUpToDate>
  <SharedDoc>false</SharedDoc>
  <HyperlinksChanged>false</HyperlinksChanged>
  <AppVersion>16.0000</AppVersion>
</Properties>
"""


def chapter_paragraphs() -> list[Paragraph]:
    return [
        ("Title", "第二章 方法框架"),
        (
            "Normal",
            "本章面向交通数字孪生与智能控制的一体化需求，构建“仿真平台—状态感知—短时预测—放行控制”的统一方法框架。研究以 SUMO 为底层仿真内核，以 TraCI 为实时交互接口，以 FastAPI 和 WebSocket 为服务层，以 Web 地图看板为展示终端，在同一闭环内完成交通运行、数据采集、预测推理和控制决策。"),
        ("Heading1", "2.1 SUMO 与 Web 结合的仿真平台搭建"),
        (
            "Normal",
            "平台搭建的目标不是单纯展示路网动画，而是形成可用于预测与控制研究的实时数字试验环境。为此，系统采用“SUMO 仿真层、TraCI 交互层、后端服务层、前端可视化层”的四层结构。SUMO 负责车辆演化与信号运行，TraCI 负责按步读取仿真状态，后端负责聚合、预测和接口封装，前端负责地图展示与交互分析。"),
        (
            "Normal",
            "在仿真对象组织上，传统以 edge 为检测单元的方法容易受到短进口道长度的影响，难以稳定表达到达、排队与放行状态。因此，本文将观测对象定义为 movement，即交叉口内由“进口道—转向—出口道”构成的转向交通流，其定义为："),
        ("Formula", "m = (i, e_in, d, e_out)"),
        (
            "Normal",
            "为避免部分进口 edge 过短带来的检测偏差，平台采用虚拟功能检测区方法。在每个 movement 上构造到达检测区、排队检测区和停止线放行检测区，并允许检测区向上游 edge 自动延伸，从而获得稳定的到达流、排队车辆数、平均速度和放行流统计量。该方法比固定单 edge 检测更能反映完整进口道状态。"),
        (
            "Normal",
            "在时间组织上，平台按 60 s 聚合一次状态，形成统一的多变量时序快照。每个快照记录 movement 的 arrival_flow、discharge_flow、mean_speed、queue_veh 等交通状态，以及 phase_id、phase_elapsed_s、green_remaining_s 等控制状态，并附加 tod_sin、tod_cos 等时间特征。该数据口径同时用于离线训练和在线推理，以保证模型输入输出的一致性。"),
        (
            "Normal",
            "在通信机制上，FastAPI 提供 REST 接口与 WebSocket 推送接口。REST 接口负责返回路网结构、预测配置和模型状态，WebSocket 则按仿真步持续推送车辆位置、信号状态、拥堵指标及预测结果。通过这种方式，仿真层产生的数据能够被上层应用低延迟消费，从而形成可观测、可交互的数字孪生平台。"),
        ("Heading1", "2.2 基于 Transformer 的短时交通流预测方法"),
        (
            "Normal",
            "短时交通流预测的任务是根据最近一段时间内的 movement 级运行状态，预测未来若干时间步的到达流量、平均速度和排队长度。设历史窗口长度为 L，预测步长为 H，则第 t 时刻的输入样本与输出目标可表示为："),
        ("Formula", "X_t = [x_{t-L+1}, ..., x_t],    Y_t = [y_{t+1}, ..., y_{t+H}]"),
        (
            "Normal",
            "本文取 L=12、H=15，即利用最近 12 个 60 s 聚合窗口预测未来 15 个窗口的交通状态。输入特征包括 arrival_flow、discharge_flow、mean_speed_mps、queue_veh、incident_flag、phase_id、phase_elapsed_s、green_remaining_s 以及全局时间特征；输出目标为 arrival_flow、mean_speed_mps 和 queue_veh。"),
        (
            "Normal",
            "在特征计算方面，平均速度采用 movement 检测区内有效车辆速度的平均值，写为："),
        ("Formula", "\\bar{v}_m = (1 / |V_m|) \\sum_{i \\in V_m} v_i"),
        (
            "Normal",
            "其中，V_m 为某一 movement 在当前聚合窗口内被检测到的车辆集合，v_i 为车辆 i 的瞬时速度。为减小量纲差异对训练稳定性的影响，输入特征在训练前按训练集统计量进行标准化处理，训练集、验证集和测试集则按运行场景的时间顺序划分，以避免不同 run 之间的信息泄漏。"),
        (
            "Normal",
            "在模型设计上，本文构建了 Transformer V1 与 Transformer V2 两类结构。二者均基于自注意力机制建模长时间依赖，其核心计算形式为："),
        ("Formula", "Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V"),
        (
            "Normal",
            "其中，Q、K、V 分别为查询、键和值矩阵，d_k 为键向量维度。自注意力机制能够根据历史状态之间的相关性自适应分配权重，适合处理具有多时间尺度波动的交通序列。最终预测映射统一表示为："),
        ("Formula", "\\hat{Y} = f_\\theta(X)"),
        (
            "Normal",
            "Transformer V1 采用扁平化时间建模思路。其做法是在每个时刻先将所有 movement 特征展平为高维向量，再映射到统一的 d_model 空间，叠加时间位置编码后送入多层 Transformer Encoder，最后取末时刻隐藏状态并通过多层感知机输出未来结果。该结构实现简单、训练稳定，是 movement 级短时预测的强基线。"),
        (
            "Normal",
            "Transformer V2 在 V1 基础上进一步引入时空分解思想。其输入首先被重组为“历史步长 × movement 数 × 每个 movement 特征数”的四维张量；随后，对每个 movement 分别进行时间编码，再在 movement 维度上执行空间编码，用于捕获同一路口、同一进口道及相邻 movement 之间的耦合影响。该结构保留了 movement 身份信息，更符合交叉口交通流的空间组织规律。"),
        (
            "Normal",
            "训练阶段采用 Huber 损失与 AdamW 优化器，以兼顾异常值鲁棒性和参数收敛稳定性；评价时综合采用 MAE、RMSE 与 WAPE，对整体误差和不同预测步长误差进行考察。"),
        ("Heading1", "2.3 面向信号放行的强化学习控制方法"),
        (
            "Normal",
            "在完成交通状态预测之后，系统进一步将预测与控制相结合，构建单路口信号放行强化学习方法。研究对象为具有多相位放行方案的交叉口，并将其抽象为马尔可夫决策过程。在每个控制时刻，智能体根据当前相位、排队状态、到达需求和可选预测信息选择动作，环境执行该动作并返回新的交通状态与奖励值。"),
        (
            "Normal",
            "状态空间采用相位级聚合设计，而非直接输入所有 movement 原始值。其核心原因在于信号控制的决策对象本质上是“当前应放行哪个相位”。因此，状态主要包括当前相位编号、当前相位已持续时间、各合法绿灯相位对应的 queue、arrival、discharge、speed 等聚合统计量，以及由交通预测模型生成的未来相位级到达和排队压力。"),
        (
            "Normal",
            "动作空间采用“保持当前主绿相位”或“切换到其他合法主绿相位”的离散集合。为满足交通工程安全约束，动作仅允许在合法主绿相位之间切换；切换过程中必须经过黄灯和全红清空阶段，并遵守最小绿灯时间与最大绿灯时间约束。该设计使强化学习决策与实际交叉口信号控制逻辑保持一致。"),
        (
            "Normal",
            "在策略学习算法上，本文采用 DQN 作为值函数近似框架。设状态为 s_t，动作为 a_t，则动作价值函数定义为："),
        ("Formula", "Q(s_t, a_t; \\theta)"),
        (
            "Normal",
            "DQN 通过近似各状态动作对的长期累积收益，引导智能体在给定状态下选择价值最大的动作，从而自动学习“何时保持绿灯、何时及时切换”的策略。"),
        (
            "Normal",
            "奖励函数以减小排队、缓解相位压力、提高通行效率并抑制无效切相为目标，其形式写为："),
        ("Formula", "r_t = -w_q Q_t - w_p P_t + w_v \\bar{v}_t - w_s I_t^{switch}"),
        (
            "Normal",
            "其中，Q_t 表示当前总排队水平，P_t 表示相位压力，\\bar{v}_t 表示平均运行速度，I_t^{switch} 表示是否发生相位切换。实际实现中，各项指标按量纲进行归一化，并加入 waiting 与 throughput 等辅助项，使奖励既能反映局部瓶颈，也能兼顾整体通行效率。"),
        (
            "Normal",
            "为对强化学习方法进行对照，本文同时引入 Webster 配时控制与 MaxPressure 放行策略作为基线。Webster 方法属于经典定时控制思想，适合提供稳定的参考方案；MaxPressure 则属于即时响应式方法，在每个决策时刻选择压力最大的相位进行放行，其决策规则可写为："),
        ("Formula", "p^* = argmax_p (Q_p + A_p)"),
        (
            "Normal",
            "该规则能够快速响应局部拥堵，是强化学习信号控制中常用的非学习基线。将 DQN 与 Webster、MaxPressure 置于统一仿真平台和统一评价指标下比较，有助于客观分析学习型控制方法的优劣。"),
        (
            "Normal",
            "需要指出的是，本文中的预测增强控制并不是将交通预测模型与强化学习策略端到端联合训练，而是采用“先预测、再控制”的串联方式。Transformer 先输出未来短时状态估计，再将其聚合为相位级前瞻特征供 DQN 使用。该设计降低了训练难度，也使预测模块和控制模块能够独立评估。"),
        ("Heading1", "2.4 本章小结"),
        (
            "Normal",
            "本章从方法层面对系统框架进行了统一说明：首先，构建了基于 SUMO、TraCI、FastAPI、WebSocket 与 Web 看板的交通仿真与交互平台，并以 movement 作为核心观测对象；其次，建立了面向 movement 级短时交通状态预测的 Transformer 方法体系；最后，在预测结果基础上构建了面向信号放行的强化学习控制框架，并设置 Webster 与 MaxPressure 作为对照基线。上述方法共同构成了后续系统实现与应用验证的理论基础。"),
    ]


def write_docx(output_path: Path) -> None:
    paragraphs = chapter_paragraphs()
    now = dt.datetime.now(dt.UTC)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", build_content_types_xml())
        zf.writestr("_rels/.rels", build_root_rels_xml())
        zf.writestr("docProps/core.xml", build_core_xml(now))
        zf.writestr("docProps/app.xml", build_app_xml())
        zf.writestr("word/document.xml", build_document_xml(paragraphs))
        zf.writestr("word/styles.xml", build_styles_xml())
        zf.writestr("word/_rels/document.xml.rels", build_document_rels_xml())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate chapter 2 Word document.")
    parser.add_argument(
        "--output",
        default="第二章_方法框架.docx",
        help="Output .docx path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    write_docx(output_path)
    print(f"generated: {output_path.resolve()}")


if __name__ == "__main__":
    main()
