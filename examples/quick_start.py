"""
快速入门示例 - 3分钟上手
===========================

本示例展示最简单的使用方式，让您在3分钟内上手文本压缩服务。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.long_text_processing_example import LongTextProcessor


def quick_start():
    """快速入门 - 最简单的使用方式"""
    
    print("="*70)
    print("快速入门：超长文本压缩与大模型交互")
    print("="*70)
    print()
    
    # 步骤1: 创建处理器（一行代码）
    print("步骤1: 创建处理器")
    print("-"*70)
    processor = LongTextProcessor(
        model_name="gpt-3.5-turbo",
        context_length=4096,
        max_output_tokens=512,
        enable_compression=True
    )
    print("✅ 处理器创建成功\n")
    
    # 步骤2: 准备超长文本
    print("步骤2: 准备超长文本")
    print("-"*70)
    long_text = """
    人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，
    它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
    
    该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。
    人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大，
    可以设想，未来人工智能带来的科技产品，将会是人类智慧的"容器"。
    
    人工智能可以对人的意识、思维的信息过程的模拟。人工智能不是人的智能，
    但能像人那样思考、也可能超过人的智能。
    
    机器学习是人工智能的核心，是使计算机具有智能的根本途径。
    深度学习是机器学习领域中一种新的方法，它源于人工神经网络的研究。
    
    卷积神经网络（CNN）是一种专门用来处理具有类似网格结构数据的神经网络。
    循环神经网络（RNN）是一种用于处理序列数据的神经网络。
    Transformer是一种基于自注意力机制的神经网络架构。
    
    大语言模型（LLM）是指具有大量参数的语言模型，通常使用Transformer架构。
    GPT（Generative Pre-trained Transformer）是一种生成式预训练Transformer模型。
    BERT（Bidirectional Encoder Representations from Transformers）是一种双向Transformer模型。
    
    自然语言处理（NLP）是人工智能的一个重要分支，它研究能实现人与计算机之间用自然语言进行有效通信的各种理论和方法。
    计算机视觉是另一个重要分支，它研究如何让计算机"看"和理解图像和视频。
    """ * 20  # 重复20次，创建超长文本
    
    print(f"文本长度: {len(long_text)} 字符")
    print("✅ 文本准备完成\n")
    
    # 步骤3: 处理并查询（一行代码）
    print("步骤3: 处理文本并查询")
    print("-"*70)
    prompt = "请总结这段关于人工智能的文字，提取关键信息。"
    
    result = processor.process_and_query(
        text=long_text,
        prompt=prompt
    )
    print("✅ 处理完成\n")
    
    # 步骤4: 查看结果
    print("步骤4: 查看结果")
    print("-"*70)
    
    if result['compression_info']:
        info = result['compression_info']
        print(f"📊 压缩统计:")
        print(f"   原始tokens: {info['original_tokens']}")
        print(f"   压缩后tokens: {info['compressed_tokens']}")
        print(f"   压缩比: {info['compression_ratio']:.2%}")
        print(f"   压缩耗时: {info['compression_time']:.3f}秒")
    else:
        print("📊 文本长度在允许范围内，无需压缩")
    
    token_info = result['token_info']
    print(f"\n📈 Token使用:")
    print(f"   原始文本tokens: {token_info['original_text_tokens']}")
    print(f"   最终输入tokens: {token_info['final_input_tokens']}")
    print(f"   上下文使用率: {token_info['context_usage']:.2%}")
    
    print("\n" + "="*70)
    print("✅ 完成！就是这么简单！")
    print("="*70)


def real_world_example():
    """真实场景示例：文档总结"""
    
    print("\n" + "="*70)
    print("真实场景示例：文档总结")
    print("="*70)
    print()
    
    # 创建处理器
    processor = LongTextProcessor(
        model_name="gpt-3.5-turbo",
        context_length=4096,
        max_output_tokens=512,
        enable_compression=True
    )
    
    # 模拟长文档
    document = """
    # 项目报告：人工智能在医疗领域的应用

    ## 1. 引言
    人工智能（AI）技术在医疗健康领域的应用日益广泛，从辅助诊断到个性化治疗，
    AI正在改变传统的医疗模式。本报告旨在分析AI在医疗领域的现状、挑战和未来趋势。

    ## 2. 主要应用领域

    ### 2.1 医学影像诊断
    AI在医学影像分析方面取得了显著进展。深度学习模型能够准确识别X光、CT、MRI等
    医学影像中的异常，辅助医生进行诊断。研究表明，AI在某些特定疾病的诊断准确率上
    已经达到或超过了人类专家的水平。

    ### 2.2 药物研发
    AI技术大大加速了新药研发的过程。通过机器学习算法，研究人员可以预测分子活性、
    优化药物结构、筛选候选药物，显著降低了研发成本和时间。

    ### 2.3 个性化治疗
    基于患者的基因数据、病史和生活习惯，AI可以为每个患者制定个性化的治疗方案。
    精准医疗的发展使得治疗效果显著提高，副作用减少。

    ### 2.4 健康管理
    可穿戴设备和健康监测APP结合AI算法，可以实时监测用户的健康状态，
    预测疾病风险，提供健康建议，实现预防性医疗。

    ## 3. 技术挑战

    ### 3.1 数据隐私与安全
    医疗数据包含敏感的个人隐私信息，如何在利用AI技术的同时保护患者隐私是一个重要挑战。
    联邦学习、差分隐私等技术为解决这一问题提供了可能。

    ### 3.2 模型可解释性
    AI模型的决策过程往往是黑盒，缺乏可解释性。在医疗领域，医生和患者需要理解
    AI的决策依据，这要求提高模型的透明度和可解释性。

    ### 3.3 法规与伦理
    AI在医疗领域的应用需要遵守严格的法规和伦理标准。如何制定合理的监管框架，
    平衡技术创新和患者安全，是政策制定者面临的重要课题。

    ## 4. 未来展望

    随着技术的不断进步，AI在医疗领域的应用将更加深入和广泛。
    我们期待看到更智能的诊断系统、更高效的药物研发流程、更个性化的治疗方案，
    以及更完善的健康管理体系。AI与医疗的融合将为人类健康事业带来革命性的变化。

    ## 5. 结论
    人工智能在医疗领域的应用前景广阔，但也面临诸多挑战。
    通过技术创新、政策引导和多方协作，我们有望克服这些挑战，
    实现AI技术在医疗健康领域的可持续发展。
    """ * 15  # 重复15次，创建超长文档
    
    print(f"文档长度: {len(document)} 字符")
    print()
    
    # 处理文档
    result = processor.process_and_query(
        text=document,
        prompt="请总结这份报告的主要内容，包括应用领域、面临的挑战和未来展望。"
    )
    
    # 显示结果
    print("="*70)
    print("处理结果")
    print("="*70)
    
    if result['compression_info']:
        info = result['compression_info']
        print(f"✅ 文档已自动压缩")
        print(f"   压缩比: {info['compression_ratio']:.2%}")
        print(f"   压缩耗时: {info['compression_time']:.3f}秒")
    
    print(f"\n📊 Token统计:")
    print(f"   原始文档: {result['token_info']['original_text_tokens']} tokens")
    print(f"   最终输入: {result['token_info']['final_input_tokens']} tokens")
    print(f"   上下文使用率: {result['token_info']['context_usage']:.2%}")
    
    print("\n" + "="*70)
    print("✅ 文档总结完成！")
    print("="*70)


def comparison_example():
    """对比示例：启用压缩 vs 不启用压缩"""
    
    print("\n" + "="*70)
    print("对比示例：启用压缩 vs 不启用压缩")
    print("="*70)
    print()
    
    # 超长文本
    long_text = "这是一个超长文本..." * 5000
    prompt = "请总结这段文字。"
    
    # 场景1: 不启用压缩
    print("场景1: 不启用压缩")
    print("-"*70)
    processor_no_compress = LongTextProcessor(
        model_name="gpt-3.5-turbo",
        context_length=4096,
        max_output_tokens=512,
        enable_compression=False  # 不启用压缩
    )
    
    result1 = processor_no_compress.process_and_query(
        text=long_text,
        prompt=prompt
    )
    
    print(f"Token使用率: {result1['token_info']['context_usage']:.2%}")
    print(f"⚠️  可能超出上下文窗口，导致截断或错误\n")
    
    # 场景2: 启用压缩
    print("场景2: 启用压缩")
    print("-"*70)
    processor_with_compress = LongTextProcessor(
        model_name="gpt-3.5-turbo",
        context_length=4096,
        max_output_tokens=512,
        enable_compression=True  # 启用压缩
    )
    
    result2 = processor_with_compress.process_and_query(
        text=long_text,
        prompt=prompt
    )
    
    if result2['compression_info']:
        info = result2['compression_info']
        print(f"✅ 文本已自动压缩")
        print(f"   压缩比: {info['compression_ratio']:.2%}")
        print(f"   Token使用率: {result2['token_info']['context_usage']:.2%}")
        print(f"   ✅ 完全符合上下文窗口要求")
    
    print("\n" + "="*70)
    print("对比总结:")
    print("  不启用压缩: 可能超出上下文窗口，导致错误")
    print("  启用压缩:  自动调整，确保符合要求")
    print("="*70)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 超长文本压缩服务 - 快速入门")
    print("="*70)
    print()
    print("本示例将展示如何在3分钟内上手文本压缩服务")
    print()
    
    # 运行示例
    quick_start()
    real_world_example()
    comparison_example()
    
    print("\n" + "="*70)
    print("🎉 恭喜！您已经掌握了基本用法！")
    print("="*70)
    print()
    print("下一步:")
    print("  1. 查看 docs/USAGE_GUIDE.md 了解更多用法")
    print("  2. 运行 examples/long_text_processing_example.py 查看完整示例")
    print("  3. 运行 examples/real_llm_integration_example.py 集成真实API")
    print()
