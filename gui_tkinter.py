import os
import sys
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import tempfile
import logging
import shutil
from src.data_processing.data_cleaner import clean_paper_data
from src.model_vector.paper_vectorizer import vectorize_paper
from src.model_vector.similarity_calculator import calculate_similarity
from src.utils import extract_core_content

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ExamSimilarityApp")


class ExamSimilarityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("试卷相似性分析系统")
        self.root.geometry("1000x800")
        self.root.configure(bg="#f0f0f0")

        # 模型映射
        self.MODEL_MAP = {
            "BERT (bert-base-chinese)": "bert",
            "BGE (bge-small-zh)": "bge",
            "Sentence-BERT (多语言)": "sentence-bert"
        }

        # 定义所有变量
        self.progress_var = tk.DoubleVar(value=0)
        self.status_var = tk.StringVar(value="就绪")
        self.model_var = tk.StringVar(value="Sentence-BERT (多语言)")
        self.method_var = tk.StringVar(value="余弦相似度与欧氏距离")
        self.threshold_var = tk.DoubleVar(value=0.7)
        self.fusion_var = tk.DoubleVar(value=0.6)
        self.type_sensitive_var = tk.BooleanVar(value=True)
        self.deduplicate_var = tk.BooleanVar(value=True)

        # 创建UI组件
        self.create_widgets()

        # 设置基础路径
        if getattr(sys, 'frozen', False):
            self.base_path = os.path.dirname(sys.executable)
        else:
            self.base_path = os.path.dirname(os.path.abspath(__file__))

        # 设置模型目录
        if getattr(sys, 'frozen', False):
            self.base_path = os.path.dirname(sys.executable)
            # 检查模型是否在可执行文件同级目录
            if os.path.exists(os.path.join(self.base_path, "models")):
                self.models_dir = os.path.join(self.base_path, "models")
            # 检查模型是否在 src/model_vector/models 目录
            elif os.path.exists(os.path.join(self.base_path, "model_vector", "models")):
                self.models_dir = os.path.join(self.base_path, "model_vector", "models")
            # 最后尝试在可执行文件同级目录的 models 目录
            else:
                self.models_dir = os.path.join(self.base_path, "models")
                logger.warning(f"模型目录不存在，尝试创建: {self.models_dir}")
                os.makedirs(self.models_dir, exist_ok=True)
        else:
            self.base_path = os.path.dirname(os.path.abspath(__file__))
            self.models_dir = os.path.join(self.base_path, "model_vector", "models")

        logger.info(f"最终模型目录: {self.models_dir}")
        logger.info(f"模型目录存在: {os.path.exists(self.models_dir)}")
        logger.info(f"模型文件: {os.listdir(self.models_dir) if os.path.exists(self.models_dir) else '无'}")

    def create_widgets(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill="both", expand=True)

        # 标题
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill="x", pady=(0, 20))
        ttk.Label(title_frame, text="试卷相似性分析系统", font=("Arial", 14, "bold")).pack(side="left")
        ttk.Label(title_frame, text="v4.0", foreground="#7f8c8d").pack(side="right")

        # 文件选择区域
        self.create_file_selection(main_frame)

        # 参数设置区域
        self.create_parameter_settings(main_frame)

        # 控制按钮
        self.create_control_buttons(main_frame)

        # 进度条
        self.create_progress_bar(main_frame)

        # 状态栏
        self.create_status_bar(main_frame)

        # 结果区域
        self.create_results_area(main_frame)

    def create_file_selection(self, parent):
        file_frame = ttk.LabelFrame(parent, text="试卷选择", padding=10)
        file_frame.pack(fill="x", pady=(0, 15))

        # 试卷A
        frame_a = ttk.Frame(file_frame)
        frame_a.pack(fill="x", pady=5)
        ttk.Label(frame_a, text="试卷A:").pack(side="left", padx=(0, 5))
        self.paper_a_entry = ttk.Entry(frame_a, width=70)
        self.paper_a_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        ttk.Button(frame_a, text="浏览...", width=10, command=self.browse_paper_a).pack(side="left")

        # 试卷B
        frame_b = ttk.Frame(file_frame)
        frame_b.pack(fill="x", pady=5)
        ttk.Label(frame_b, text="试卷B:").pack(side="left", padx=(0, 5))
        self.paper_b_entry = ttk.Entry(frame_b, width=70)
        self.paper_b_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        ttk.Button(frame_b, text="浏览...", width=10, command=self.browse_paper_b).pack(side="left")

        # 输出位置
        frame_output = ttk.Frame(file_frame)
        frame_output.pack(fill="x", pady=5)
        ttk.Label(frame_output, text="输出位置:").pack(side="left", padx=(0, 5))
        self.output_entry = ttk.Entry(frame_output, width=70)
        self.output_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        ttk.Button(frame_output, text="浏览...", width=10, command=self.browse_output).pack(side="left")

    def create_parameter_settings(self, parent):
        param_frame = ttk.LabelFrame(parent, text="分析参数", padding=10)
        param_frame.pack(fill="x", pady=(0, 15))

        # 模型选择
        model_frame = ttk.Frame(param_frame)
        model_frame.pack(fill="x", pady=5)
        ttk.Label(model_frame, text="向量化模型:").pack(side="left", padx=(0, 10))
        model_combo = ttk.Combobox(
            model_frame,
            textvariable=self.model_var,
            values=("BERT (bert-base-chinese)", "BGE (bge-small-zh)", "Sentence-BERT (多语言)"),
            state="readonly",
            width=25
        )
        model_combo.pack(side="left")

        # 方法选择
        ttk.Label(model_frame, text="相似度方法:").pack(side="left", padx=(30, 10))
        method_combo = ttk.Combobox(
            model_frame,
            textvariable=self.method_var,
            values=("余弦相似度与欧氏距离",),
            state="readonly",
            width=20
        )
        method_combo.pack(side="left")

        # 阈值设置
        threshold_frame = ttk.Frame(param_frame)
        threshold_frame.pack(fill="x", pady=5)
        ttk.Label(threshold_frame, text="相似度阈值:").pack(side="left", padx=(0, 10))

        # 阈值滑块
        threshold_scale = ttk.Scale(
            threshold_frame,
            from_=0.1,  # 浮点数
            to=1.0,  # 浮点数
            variable=self.threshold_var,
            length=200,
            command=self.update_threshold_display
        )
        threshold_scale.pack(side="left")

        # 阈值显示
        self.threshold_display = ttk.Label(threshold_frame, text="0.70", width=4)
        self.threshold_display.pack(side="left", padx=(10, 30))

        # 类型匹配
        ttk.Checkbutton(
            threshold_frame,
            text="要求题目类型匹配",
            variable=self.type_sensitive_var
        ).pack(side="left")

        # 融合权重
        fusion_frame = ttk.Frame(param_frame)
        fusion_frame.pack(fill="x", pady=5)
        ttk.Label(fusion_frame, text="余弦权重:").pack(side="left", padx=(0, 10))

        # 融合权重滑块 - 修正部分
        fusion_scale = ttk.Scale(
            fusion_frame,
            from_=0.0,  # 修正为浮点数
            to=1.0,  # 修正为浮点数
            variable=self.fusion_var,
            length=200,
            command=self.update_fusion_display
        )
        fusion_scale.pack(side="left")

        # 权重显示
        self.fusion_display = ttk.Label(fusion_frame, text="0.60", width=4)
        self.fusion_display.pack(side="left", padx=(10, 30))

        # 去重选项
        deduplicate_frame = ttk.Frame(param_frame)
        deduplicate_frame.pack(fill="x", pady=5)
        ttk.Checkbutton(
            deduplicate_frame,
            text="在比对前进行题目去重",
            variable=self.deduplicate_var
        ).pack(side="left")

    def create_control_buttons(self, parent):
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill="x", pady=(0, 15))

        ttk.Button(
            btn_frame,
            text="开始分析",
            command=self.start_analysis,
            width=15
        ).pack(side="left", padx=10)

        ttk.Button(
            btn_frame,
            text="清除结果",
            command=self.clear_results,
            width=15
        ).pack(side="left", padx=10)

        ttk.Button(
            btn_frame,
            text="退出系统",
            command=self.root.quit,
            width=15
        ).pack(side="right", padx=10)

    def create_progress_bar(self, parent):
        bar_frame = ttk.Frame(parent)
        bar_frame.pack(fill="x", pady=(0, 10))

        self.progress_bar = ttk.Progressbar(
            bar_frame,
            variable=self.progress_var,
            maximum=100,
            length=500
        )
        self.progress_bar.pack(fill="x")

    def create_status_bar(self, parent):
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(
            status_frame,
            textvariable=self.status_var,
            anchor="w"
        ).pack(fill="x")

    def create_results_area(self, parent):
        result_frame = ttk.LabelFrame(parent, text="分析结果", padding=10)
        result_frame.pack(fill="both", expand=True)

        # 创建选项卡
        nb = ttk.Notebook(result_frame)
        nb.pack(fill="both", expand=True, padx=5, pady=5)

        # 摘要选项卡
        summary_frame = ttk.Frame(nb)
        nb.add(summary_frame, text="摘要")

        self.summary_text = scrolledtext.ScrolledText(
            summary_frame,
            wrap=tk.WORD,
            font=("Arial", 10),
            padx=10, pady=10
        )
        self.summary_text.pack(fill="both", expand=True)
        self.summary_text.config(state="disabled")

        # 详情选项卡
        detail_frame = ttk.Frame(nb)
        nb.add(detail_frame, text="详细结果")

        self.detail_text = scrolledtext.ScrolledText(
            detail_frame,
            wrap=tk.WORD,
            font=("Arial", 10),
            padx=10, pady=10
        )
        self.detail_text.pack(fill="both", expand=True)
        self.detail_text.config(state="disabled")

        # 配置文本标签
        self.summary_text.tag_configure("header", font=("Arial", 12, "bold"), foreground="#2c3e50")
        self.summary_text.tag_configure("subheader", font=("Arial", 11, "bold"))
        self.detail_text.tag_configure("header", font=("Arial", 12, "bold"), foreground="#2c3e50")
        self.detail_text.tag_configure("highlight", font=("Arial", 11, "bold"), foreground="#e74c3c")

    def update_threshold_display(self, event=None):
        self.threshold_display.config(text=f"{self.threshold_var.get():.2f}")

    def update_fusion_display(self, event=None):
        self.fusion_display.config(text=f"{self.fusion_var.get():.2f}")

    def browse_paper_a(self):
        path = filedialog.askopenfilename(
            title="选择试卷A",
            filetypes=[("JSON文件", "*.json")]
        )
        if path:
            self.paper_a_entry.delete(0, tk.END)
            self.paper_a_entry.insert(0, path)

    def browse_paper_b(self):
        path = filedialog.askopenfilename(
            title="选择试卷B",
            filetypes=[("JSON文件", "*.json")]
        )
        if path:
            self.paper_b_entry.delete(0, tk.END)
            self.paper_b_entry.insert(0, path)

    def browse_output(self):
        path = filedialog.asksaveasfilename(
            title="保存结果",
            filetypes=[("JSON文件", "*.json")],
            defaultextension=".json"
        )
        if path:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, path)

    def clear_results(self):
        self.summary_text.config(state="normal")
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.config(state="disabled")

        self.detail_text.config(state="normal")
        self.detail_text.delete(1.0, tk.END)
        self.detail_text.config(state="disabled")

        self.progress_var.set(0)
        self.status_var.set("结果已清除")

    def update_progress(self, value, message=None):
        self.progress_var.set(value)
        if message:
            self.status_var.set(message)
        self.root.update_idletasks()  # 确保UI更新

    def start_analysis(self):
        # 验证输入
        paper_a = self.paper_a_entry.get()
        paper_b = self.paper_b_entry.get()
        output_path = self.output_entry.get()

        if not all([paper_a, paper_b, output_path]):
            messagebox.showerror("错误", "请填写所有字段!")
            return

        if not (os.path.exists(paper_a) and os.path.exists(paper_b)):
            messagebox.showerror("错误", "文件不存在!")
            return

        try:
            # 创建临时目录
            temp_dir = tempfile.mkdtemp(prefix="exam_similarity_")
            self.update_progress(5, f"创建临时目录: {temp_dir}")

            # 处理试卷A
            cleaned_a = os.path.join(temp_dir, "paper_a_cleaned.json")
            vector_a = os.path.join(temp_dir, "paper_a_vectorized.json")

            self.update_progress(10, "清洗试卷A...")
            clean_paper_data(paper_a, cleaned_a, paper_id="paper_a", deduplicate=self.deduplicate_var.get())

            self.update_progress(30, "向量化试卷A...")
            vectorize_paper(cleaned_a, vector_a, self.MODEL_MAP[self.model_var.get()], model_dir=self.models_dir)

            # 处理试卷B
            cleaned_b = os.path.join(temp_dir, "paper_b_cleaned.json")
            vector_b = os.path.join(temp_dir, "paper_b_vectorized.json")

            self.update_progress(40, "清洗试卷B...")
            clean_paper_data(paper_b, cleaned_b, paper_id="paper_b", deduplicate=self.deduplicate_var.get())

            self.update_progress(60, "向量化试卷B...")
            vectorize_paper(cleaned_b, vector_b, self.MODEL_MAP[self.model_var.get()], model_dir=self.models_dir)

            # 计算相似度
            self.update_progress(70, "计算相似度...")
            results = calculate_similarity(
                vector_a,
                vector_b,
                threshold=self.threshold_var.get(),
                type_sensitive=self.type_sensitive_var.get(),
                fusion_weight=self.fusion_var.get(),
                deduplicate=self.deduplicate_var.get()
            )

            # 保存结果
            self.update_progress(90, "保存结果...")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            # 显示结果
            self.update_progress(95, "生成结果报告...")
            self.display_results(results)
            self.update_progress(100, f"分析完成! 结果已保存到: {output_path}")

        except Exception as e:
            messagebox.showerror("错误", f"处理过程中发生错误: {str(e)}")
            logger.exception("分析过程中出错")
            self.update_progress(0, f"错误: {str(e)}")
        finally:
            # 清理临时文件
            try:
                if 'temp_dir' in locals() and os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    self.status_var.set(f"{self.status_var.get()} | 临时文件已清理")
            except Exception as e:
                logger.warning(f"清理临时文件失败: {str(e)}")

    def display_results(self, results):
        # 准备摘要
        self.summary_text.config(state="normal")
        self.summary_text.delete(1.0, tk.END)

        total_pairs = results.get("total_pairs", 0)
        if total_pairs == 0:
            self.summary_text.insert(tk.END, "未找到相似题目对", "header")
        else:
            self.summary_text.insert(tk.END, "相似度分析结果摘要\n", "header")
            self.summary_text.insert(tk.END, "=" * 80 + "\n")
            self.summary_text.insert(tk.END, f"试卷A: {os.path.basename(results['paper_a'])}\n")
            self.summary_text.insert(tk.END, f"试卷B: {os.path.basename(results['paper_b'])}\n")
            self.summary_text.insert(tk.END, f"融合权重(余弦): {results.get('fusion_weight', 0.6):.2f}\n")
            self.summary_text.insert(tk.END, f"相似度阈值: {results['threshold']:.2f}\n")
            self.summary_text.insert(tk.END, f"类型匹配要求: {'是' if results['type_sensitive'] else '否'}\n")
            self.summary_text.insert(tk.END, f"去重处理: {'是' if results.get('deduplicate', True) else '否'}\n")
            self.summary_text.insert(tk.END, f"共找到 {total_pairs} 对相似题目\n")
            self.summary_text.insert(tk.END, "=" * 80 + "\n\n")

            if results["similar_pairs"]:
                top = results["similar_pairs"][0]
                self.summary_text.insert(tk.END, "最高相似度题目对:\n", "subheader")
                self.summary_text.insert(tk.END, f"相似度: {top['similarity']:.4f}\n")
                self.summary_text.insert(tk.END, f"试卷A题号: {top['paper_a']['id']}\n")
                self.summary_text.insert(tk.END, f"试卷B题号: {top['paper_b']['id']}\n")
                self.summary_text.insert(tk.END, f"试卷A题型: {top['paper_a']['type']}\n")
                self.summary_text.insert(tk.END, f"试卷B题型: {top['paper_b']['type']}\n")

        self.summary_text.config(state="disabled")

        # 准备详情
        self.detail_text.config(state="normal")
        self.detail_text.delete(1.0, tk.END)

        if total_pairs == 0:
            self.detail_text.insert(tk.END, "未找到相似题目对", "header")
        else:
            self.detail_text.insert(tk.END, f"相似题目对详情 (共{total_pairs}对)\n", "header")
            self.detail_text.insert(tk.END, "=" * 100 + "\n\n")

            for idx, pair in enumerate(results["similar_pairs"], 1):
                text_a = extract_core_content(pair['paper_a']['text'])
                text_b = extract_core_content(pair['paper_b']['text'])

                self.detail_text.insert(tk.END,
                                        f"【第{idx}对】相似度: {pair['similarity']:.4f} "
                                        f"(A题号:{pair['paper_a']['id']} ↔ B题号:{pair['paper_b']['id']})\n",
                                        "highlight"
                                        )
                self.detail_text.insert(tk.END, f"试卷A[{pair['paper_a']['type']}]:\n{text_a}\n\n")
                self.detail_text.insert(tk.END, f"试卷B[{pair['paper_b']['type']}]:\n{text_b}\n\n")
                self.detail_text.insert(tk.END, "-" * 100 + "\n\n")

        self.detail_text.config(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    app = ExamSimilarityApp(root)
    root.mainloop()