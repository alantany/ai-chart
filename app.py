import streamlit as st
import pandas as pd
import plotly.express as px
import sqlite3
import plotly.graph_objects as go
import numpy as np

def main():
    st.set_page_config(layout="wide")
    st.title("通用数据分析平台")
    
    # 文件上传
    uploaded_files = st.file_uploader("上传CSV文件", type="csv", accept_multiple_files=True)
    
    # 数据库连接（示例）
    db_connection = st.text_input("输入数据库连接字符串（示例：sqlite:///example.db）")
    
    if uploaded_files or db_connection:
        data_sources = process_data_sources(uploaded_files, db_connection)
        if data_sources:
            perform_analysis(data_sources)
            perform_advanced_analysis(data_sources)

def process_data_sources(uploaded_files, db_connection):
    data_sources = {}
    
    # 处理上传的CSV文件
    for file in uploaded_files:
        df = pd.read_csv(file)
        data_sources[file.name] = df
    
    # 处理数据库连接
    if db_connection:
        try:
            conn = sqlite3.connect(db_connection.split("://")[-1])
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            for table in tables:
                df = pd.read_sql_query(f"SELECT * FROM {table[0]}", conn)
                data_sources[table[0]] = df
            conn.close()
        except Exception as e:
            st.error(f"连接数据库时出错：{str(e)}")
    
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
                                                ["热力图", "条形图", "散点矩阵图", "平行坐标图"])
            
            if multi_dim_chart_type == "热力图":
                fig = px.imshow(pivot_table, text_auto=True, aspect="auto")
                st.plotly_chart(fig, use_container_width=True)
            
            elif multi_dim_chart_type == "条形图":
                fig = px.bar(pivot_table.reset_index(), x=dimensions[0], y=measures[0],
                             color=dimensions[1] if len(dimensions) > 1 else None,
                             barmode='group')
                st.plotly_chart(fig, use_container_width=True)
            
            elif multi_dim_chart_type == "散点矩阵图":
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
                corr_chart_type = st.selectbox("选择相关性分析图表类型", 
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

if __name__ == "__main__":
    main()