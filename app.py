import streamlit as st
import pandas as pd
import plotly.express as px
import sqlite3
import plotly.graph_objects as go
import numpy as np
import openpyxl
from openai import OpenAI

def main():
    st.set_page_config(layout="wide")
    st.title("通用数据分析平台")
    
    st.write("支持的文件类型：CSV (.csv), Excel (.xlsx), SQLite 数据库 (.db 或 .sqlite)")
    
    # 使用 st.session_state 来存储上传的文件
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = None
    
    # 文件上传组件
    uploaded_files = st.file_uploader("上传数据文件", type=["csv", "xlsx", "db", "sqlite"], accept_multiple_files=False)
    
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
    
    if st.session_state.uploaded_files:
        data_sources = process_data_sources(st.session_state.uploaded_files)
        if data_sources:
            analysis_type = st.sidebar.selectbox(
                "选择分析类型",
                ["基础分析", "高级分析", "自然语言分析"]
            )
            
            if analysis_type == "基础分析":
                perform_analysis(data_sources)
            elif analysis_type == "高级分析":
                perform_advanced_analysis(data_sources)
            else:
                perform_nlp_analysis(data_sources)

def process_data_sources(uploaded_file):
    data_sources = {}
    
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            data_sources[uploaded_file.name] = df
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
            data_sources[uploaded_file.name] = df
        elif uploaded_file.name.endswith(('.db', '.sqlite')):
            # 将上传的文件保存到临时文件
            with open("temp_db.sqlite", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            conn = sqlite3.connect("temp_db.sqlite")
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            if tables:
                for table in tables:
                    df = pd.read_sql_query(f"SELECT * FROM {table[0]}", conn)
                    data_sources[f"{uploaded_file.name} - {table[0]}"] = df
            else:
                st.warning(f"数据库 {uploaded_file.name} 中没有找到任何表")
            conn.close()
        else:
            st.warning(f"不支持的文件类型：{uploaded_file.name}")
    except Exception as e:
        st.error(f"处理文件 {uploaded_file.name} 时出错：{str(e)}")
    
    return data_sources

def perform_analysis(data_sources):
    st.header("数据分析")
    
    # 选择数据源
    selected_source = st.selectbox("选择数据源", list(data_sources.keys()))
    df = data_sources[selected_source]
    
    # 显示数据预览
    st.subheader("数据预览")
    st.dataframe(df.head())
    
    # 字段选择
    st.subheader("字段选择")
    columns = df.columns.tolist()
    
    # 使用 multiselect 来选择 X 轴和 Y 轴
    x_axis = st.multiselect("选择X轴字段", columns, key="x_axis_select")
    y_axis = st.multiselect("选择Y轴字段", columns, key="y_axis_select")
    
    # 图表类型选择
    chart_types = [
        "折线图", "柱状图", "散点图", "面积图", "箱线图", "小提琴图",
        "直方图", "密度图", "热力图", "等高线图", "极坐标图", "树状图",
        "旭日图", "漏斗图", "瀑布图"
    ]
    chart_type = st.selectbox("选择图表类型", chart_types)
    
    # 绘制图表
    if x_axis and y_axis:
        try:
            # 使用选择的第一个字段作为 X 轴和 Y 轴
            x = x_axis[0]
            y = y_axis[0]
            
            # 绘制图表的代码保持不变
            if chart_type == "折线图":
                fig = px.line(df, x=x, y=y)
            elif chart_type == "柱状图":
                fig = px.bar(df, x=x, y=y)
            elif chart_type == "散点图":
                fig = px.scatter(df, x=x, y=y)
            elif chart_type == "面积图":
                fig = px.area(df, x=x, y=y)
            elif chart_type == "箱线图":
                fig = px.box(df, x=x, y=y)
            elif chart_type == "小提琴图":
                fig = px.violin(df, x=x, y=y)
            elif chart_type == "直方图":
                fig = px.histogram(df, x=x)
            elif chart_type == "密度图":
                fig = px.density_contour(df, x=x, y=y)
            elif chart_type == "热力图":
                fig = px.density_heatmap(df, x=x, y=y)
            elif chart_type == "等高线图":
                fig = px.density_contour(df, x=x, y=y)
            elif chart_type == "极坐标图":
                fig = px.line_polar(df, r=y, theta=x, line_close=True)
            elif chart_type == "树状图":
                fig = px.treemap(df, path=[x], values=y)
            elif chart_type == "旭日图":
                fig = px.sunburst(df, path=[x], values=y)
            elif chart_type == "漏斗图":
                fig = px.funnel(df, x=x, y=y)
            elif chart_type == "瀑布图":
                # 使用 plotly.graph_objects 创建瀑布图
                sorted_df = df.sort_values(by=x)
                fig = go.Figure(go.Waterfall(
                    name="瀑布图", orientation="v",
                    measure=["relative"] * len(sorted_df),
                    x=sorted_df[x],
                    y=sorted_df[y],
                    connector={"line": {"color": "rgb(63, 63, 63)"}},
                ))
                fig.update_layout(title=f"{y} 的瀑布图", showlegend=False)
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"绘制图表时出错：{str(e)}")
    else:
        st.warning("请选择X轴和Y轴字段")
    
    # 基本统计信息
    st.subheader("基本统计信息")
    st.write(df.describe())

def perform_advanced_analysis(data_sources):
    st.header("高级数据分析")
    
    selected_source = st.selectbox("选择数据源", list(data_sources.keys()), key="advanced_source")
    df = data_sources[selected_source]
    
    columns = df.columns.tolist()
    
    # 多维度分析
    st.subheader("多维度分析")
    
    dimensions = st.multiselect("选择维度", columns, key="dimensions_select")
    measures = st.multiselect("选择度量", columns, key="measures_select")
    
    if dimensions and measures:
        try:
            # 创建透视表
            pivot_table = pd.pivot_table(df, values=measures, 
                                         index=dimensions, 
                                         aggfunc=lambda x: x.mode().iloc[0] if x.dtype == 'object' else x.mean())
            st.write(pivot_table)
            
            # 多维度分析展示方式选择
            multi_dim_chart_type = st.selectbox("选择多维度分析图表类型", 
                                                ["热力图", "条形图", "散点矩阵图", "行坐标图"])
            
            if multi_dim_chart_type == "热力图":
                fig = px.imshow(pivot_table, text_auto=True, aspect="auto")
                st.plotly_chart(fig, use_container_width=True)
            
            elif multi_dim_chart_type == "条形图":
                fig = px.bar(pivot_table.reset_index(), x=dimensions[0], y=measures[0],
                             color=dimensions[1] if len(dimensions) > 1 else None,
                             barmode='group')
                st.plotly_chart(fig, use_container_width=True)
            
            elif multi_dim_chart_type == "散点矩阵":
                fig = px.scatter_matrix(df[dimensions + measures])
                st.plotly_chart(fig, use_container_width=True)
            
            elif multi_dim_chart_type == "平行坐标图":
                fig = px.parallel_coordinates(df, dimensions=dimensions + measures)
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"创建多维度分析图表时出错：{str(e)}")
    else:
        st.warning("请至少选择一个维度和一个度量")
    
    # 相关性分析
    st.subheader("相关性分析")
    correlation_columns = st.multiselect("选择要分析相关性的列", columns, key="correlation_select")

    if len(correlation_columns) >= 2:
        try:
            # 只选择数值型列进行相关性分析
            numeric_columns = df[correlation_columns].select_dtypes(include=[np.number]).columns
            if len(numeric_columns) < 2:
                st.warning("请至少选择两个数值型列进行相关性分析")
            else:
                correlation_matrix = df[numeric_columns].corr()
                
                # 相关性分析展示方式选择
                corr_chart_type = st.selectbox("选择相关分析图表类型", 
                                               ["热力图", "散点矩阵图"])
                
                if corr_chart_type == "热力图":
                    fig = px.imshow(correlation_matrix, text_auto=True, zmin=-1, zmax=1,
                                    color_continuous_scale=px.colors.diverging.RdBu_r)
                    st.plotly_chart(fig, use_container_width=True)
                
                elif corr_chart_type == "散点矩阵图":
                    fig = px.scatter_matrix(df[numeric_columns])
                    st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"进行相关性分析时出错：{str(e)}")
    else:
        st.warning("请至少选择两列进行相关性分析")

def perform_nlp_analysis(data_sources):
    st.header("自然语言分析")
    
    selected_source = st.selectbox("选择数据源", list(data_sources.keys()), key="nlp_source")
    df = data_sources[selected_source]
    
    # 获取实际的表名
    if selected_source.endswith(('.xlsx', '.csv')):
        table_name = 'data'  # 为Excel和CSV文件使用固定的表名
    else:
        table_name = selected_source.split(" - ")[-1] if " - " in selected_source else selected_source
    
    # 显示数据预览
    st.subheader("数据预览")
    st.dataframe(df.head())
    
    # 自然语言输入
    nl_query = st.text_input("输入你的自然语言查询")
    
    if nl_query:
        try:
            # 调用OpenAI API转换自然语言到SQL
            sql_query = nl_to_sql(nl_query, df, table_name)
            
            st.subheader("生成的SQL查询")
            st.code(sql_query, language="sql")
            
            # 执行SQL查询
            result = execute_sql_query(df, sql_query, table_name)
            
            st.subheader("查询结果")
            st.dataframe(result)
            
            # 生成可视化
            if not result.empty:
                st.subheader("数据可视化")
                
                # 初始化 session state
                if 'chart_type' not in st.session_state:
                    st.session_state.chart_type = "柱状图"
                if 'x_axis' not in st.session_state:
                    st.session_state.x_axis = []
                if 'y_axis' not in st.session_state:
                    st.session_state.y_axis = []
                if 'color' not in st.session_state:
                    st.session_state.color = "无"
                
                # 使用 st.form 来组织所有的可视化选项
                with st.form("visualization_form"):
                    chart_type = st.selectbox("选择图表类型", [
                        "柱状图", "折线图", "散点图", "饼图", "面积图", "箱线图", "小提琴图", 
                        "直方图", "密度图", "热力图", "等高线图", "极坐标图", "树状图", 
                        "旭日图", "漏斗图", "瀑布图", "甘特图"
                    ], index=[
                        "柱状图", "折线图", "散点图", "饼图", "面积图", "箱线图", "小提琴图", 
                        "直方图", "密度图", "热力图", "等高线图", "极坐标图", "树状图", 
                        "旭日图", "漏斗图", "瀑布图", "甘特图"
                    ].index(st.session_state.chart_type))

                    numeric_columns = result.select_dtypes(include=[np.number]).columns
                    all_columns = result.columns

                    x_axis = st.multiselect("选择 X 轴（可多选）", all_columns, default=st.session_state.x_axis)
                    y_axis = st.multiselect("选择 Y 轴（可多选）", numeric_columns, default=st.session_state.y_axis)
                    color = st.selectbox("选择颜色分组（可选）", ["无"] + list(all_columns), index=["无"] + list(all_columns).index(st.session_state.color) if st.session_state.color in all_columns else 0)

                    # 提交按钮
                    submitted = st.form_submit_button("生成图表")

                # 只有在提交表单后才执行图表生成
                if submitted:
                    st.session_state.chart_type = chart_type
                    st.session_state.x_axis = x_axis
                    st.session_state.y_axis = y_axis
                    st.session_state.color = color

                    color = None if color == "无" else color
                    
                    if not x_axis or not y_axis:
                        st.warning("请至少选择一个 X 轴和一个 Y 轴字段。")
                    elif check_chart_compatibility(chart_type, x_axis, y_axis, color):
                        fig = generate_chart(result, chart_type, x_axis, y_axis, color)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(get_chart_suggestion(chart_type, x_axis, y_axis, color))
                
                # 显示当前选择的值（用于调试）
                st.write("当前选择：")
                st.write(f"图表类型: {st.session_state.chart_type}")
                st.write(f"X 轴: {st.session_state.x_axis}")
                st.write(f"Y 轴: {st.session_state.y_axis}")
                st.write(f"颜色分组: {st.session_state.color}")

        except Exception as e:
            st.error(f"处理查询时出错：{str(e)}")

def check_chart_compatibility(chart_type, x, y, color):
    if not x or not y:
        return False
    if chart_type in ["散点图", "热力图", "饼图", "漏斗图"] and (len(x) != 1 or len(y) != 1):
        return False
    if chart_type in ["树状图", "旭日图"] and len(x) < 2:
        return False
    return True

def get_chart_suggestion(chart_type, x, y, color):
    # 根据当前选择的字段给出图表类型建议
    if not x and not y:
        return "请至少选择一个 X 轴和一个 Y 轴字段。"
    elif len(x) == 1 and len(y) == 1:
        return f"对于单个 X 和 Y 轴，您可以尝试使用：散点图、柱状图、折线图或饼图。"
    elif len(x) > 1 and len(y) >= 1:
        return f"对于多个维度，您可以尝试使用：树状图、旭日图或平行坐标图。"
    elif len(y) > 1:
        return f"对于多个度量值，您可以尝试使用：柱状图（分组）、折线图（多线）或面积图（堆叠）。"
    else:
        return f"当前选择的字段组合不适合 {chart_type}。请调整您的选择或尝试其他图表类型。"

def generate_chart(df, chart_type, x, y, color):
    if df.empty:
        return None
    
    try:
        if chart_type == "柱状图":
            return px.bar(df, x=x[0], y=y, color=color, barmode='group')
        elif chart_type == "折线图":
            return px.line(df, x=x[0], y=y, color=color)
        elif chart_type == "散点图":
            return px.scatter(df, x=x[0], y=y[0], color=color)
        elif chart_type == "面积图":
            return px.area(df, x=x[0], y=y, color=color)
        elif chart_type == "箱线图":
            return px.box(df, x=x[0], y=y[0], color=color)
        elif chart_type == "小提琴图":
            return px.violin(df, x=x[0], y=y[0], color=color)
        elif chart_type == "热力图":
            if len(df.columns) > 2:
                corr_df = df[df.select_dtypes(include=[np.number]).columns].corr()
                return px.imshow(corr_df, text_auto=True, aspect="auto")
            else:
                return px.density_heatmap(df, x=x[0], y=y[0])
        elif chart_type == "树状图":
            return px.treemap(df, path=x + y, values=y[-1] if y else None)
        elif chart_type == "旭日图":
            return px.sunburst(df, path=x + y, values=y[-1] if y else None)
        elif chart_type == "平行坐标图":
            return px.parallel_coordinates(df, dimensions=x + y, color=color)
        elif chart_type == "散点矩阵图":
            return px.scatter_matrix(df, dimensions=x + y, color=color)
        elif chart_type == "饼图":
            return px.pie(df, names=x[0], values=y[0])
        elif chart_type == "直方图":
            return px.histogram(df, x=x[0], y=y[0] if y else None, color=color)
        elif chart_type == "密度图":
            return px.density_contour(df, x=x[0], y=y[0], color=color)
        elif chart_type == "等高线图":
            return px.density_contour(df, x=x[0], y=y[0], color=color)
        elif chart_type == "极坐标图":
            return px.line_polar(df, r=y[0], theta=x[0], color=color)
        elif chart_type == "漏斗图":
            return px.funnel(df, x=x[0], y=y[0])
        elif chart_type == "瀑布图":
            sorted_df = df.sort_values(by=x[0])
            fig = go.Figure(go.Waterfall(
                name="瀑布图", orientation="v",
                measure=["relative"] * len(sorted_df),
                x=sorted_df[x[0]],
                y=sorted_df[y[0]],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
            ))
            fig.update_layout(title=f"{y[0]} 的瀑布图", showlegend=False)
            return fig
        # 甘特图需要特殊处理，这里暂时省略
    except Exception as e:
        st.error(f"生成图表时出错：{str(e)}")
        return None
    
    return None

def nl_to_sql(nl_query, df, table_name):
    client = OpenAI(
        api_key="sk-1pUmQlsIkgla3CuvKTgCrzDZ3r0pBxO608YJvIHCN18lvOrn",
        base_url="https://api.chatanywhere.tech/v1"
    )
    
    # 准备表结构信息
    table_info = f"表名: {table_name}\n表结构:\n"
    for column, dtype in df.dtypes.items():
        table_info += f"{column}: {dtype}\n"
    
    # 准备提示
    prompt = f"""
    给定以下表结构:
    {table_info}
    
    将以下自然语言查询转换为SQL:
    {nl_query}
    
    请使用 '{table_name}' 作为表名。只返回SQL查询，不要有任何其他解释或格式标记。
    """
    
    # 调用API
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a SQL expert. Convert natural language queries to SQL."},
            {"role": "user", "content": prompt}
        ]
    )
    
    # 获取生成的SQL并清理
    sql = response.choices[0].message.content.strip()
    
    # 除可能的 Markdown 代码块标记
    sql = sql.replace("```sql", "").replace("```", "").strip()
    
    return sql

def execute_sql_query(df, query, table_name):
    try:
        import sqlite3
        import pandas as pd
        
        conn = sqlite3.connect(':memory:')
        df.to_sql(table_name, conn, index=False)
        
        # 替换查询中的文件名为表名
        query = query.replace(table_name + '.xlsx', table_name)
        query = query.replace(table_name + '.csv', table_name)
        
        # 打印实际执行的SQL查询，用于调试
        st.write("执行的SQL查询：", query)
        
        # 执行查询
        result = pd.read_sql_query(query, conn)
        
        conn.close()
        return result
    except Exception as e:
        st.error(f"执行SQL查询时出错：{str(e)}")
        return pd.DataFrame()

if __name__ == "__main__":
    main()