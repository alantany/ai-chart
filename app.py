import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import sqlite3
import io
from openai import OpenAI
import plotly.express as px
import plotly.graph_objects as go

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key="sk-1pUmQlsIkgla3CuvKTgCrzDZ3r0pBxO608YJvIHCN18lvOrn",
    base_url="https://api.chatanywhere.tech/v1"
)

# 设置页面标题
st.set_page_config(page_title="基于AI的数据分析", layout="wide")

# 主页面标题
st.title("基于AI的数据分析")

# NL2SQL函数
def nl_to_sql(nl_query, table_info):
    prompt = f"""
    给定以下表格信息：
    {table_info}
    
    将以下自然语言查询转换为SQL：
    {nl_query}
    
    只返回SQL查询，不要包含任何解释。
    """
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "你是一个SQL专家，能够将自然语言查询转换为SQL语句。请用中文回答。"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.5,
    )
    
    return response.choices[0].message.content.strip()

def execute_sql_query(df, sql_query):
    # 创建一个临时的 SQLite 数据库在内存中
    engine = create_engine('sqlite:///:memory:')
    
    # 将 DataFrame 写入 SQLite 数据库
    df.to_sql('data', engine, index=False)
    
    # 执行 SQL 查询
    result = pd.read_sql_query(sql_query, engine)
    
    return result

# 数据加载函数
def load_data():
    uploaded_file = st.file_uploader("上传Excel文件", type=["xlsx", "xls"])
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        return df
    return None

# 加载数据
df = load_data()

if df is not None:
    st.success("文件已成功加载！")
    
    # 显示数据预览
    st.subheader("数据预览")
    st.dataframe(df.head())
    
    # 生成表信息
    table_info = f"Table name: data\nColumns: {', '.join(df.columns)}\n"
    table_info += "\n".join([f"{col}: {df[col].dtype}" for col in df.columns])
    
    # 自然语言查询
    nl_query = st.text_input("请输入您的自然语言查询：")
    
    if nl_query:
        # 使用NL2SQL转换查询
        sql_query = nl_to_sql(nl_query, table_info)
        
        st.write(f"生成的SQL查询：{sql_query}")
        
        # 执行SQL查询
        try:
            result_df = execute_sql_query(df, sql_query)
            st.write("查询结果：")
            st.dataframe(result_df)
            
            # 分析选项
            if not result_df.empty:
                st.subheader("数据可视化")
                
                # 创建两列布局
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    analysis_type = st.selectbox("选择分析图表类型", [
                        "柱状图", "折线图", "散点图", "饼图", "箱线图", "热力图", "面积图", "直方图"
                    ])
                    
                    x_column = st.selectbox("选择X轴", result_df.columns)
                    y_column = st.selectbox("选择Y轴", result_df.columns)
                    
                    if analysis_type in ["散点图", "热力图"]:
                        color_column = st.selectbox("选择颜色映射列", result_df.columns)
                    
                    # 图表大小调整
                    chart_width = st.slider("图表宽度", 400, 1200, 800)
                    chart_height = st.slider("图表高度", 300, 900, 500)
                
                with col2:
                    if analysis_type == "柱状图":
                        fig = px.bar(result_df, x=x_column, y=y_column, title="柱状图")
                    elif analysis_type == "折线图":
                        fig = px.line(result_df, x=x_column, y=y_column, title="折线图")
                    elif analysis_type == "散点图":
                        fig = px.scatter(result_df, x=x_column, y=y_column, color=color_column, title="散点图")
                    elif analysis_type == "饼图":
                        fig = px.pie(result_df, values=y_column, names=x_column, title="饼图")
                    elif analysis_type == "箱线图":
                        fig = px.box(result_df, x=x_column, y=y_column, title="箱线图")
                    elif analysis_type == "热力图":
                        fig = px.density_heatmap(result_df, x=x_column, y=y_column, z=color_column, title="热力图")
                    elif analysis_type == "面积图":
                        fig = px.area(result_df, x=x_column, y=y_column, title="面积图")
                    elif analysis_type == "直方图":
                        fig = px.histogram(result_df, x=x_column, title="直方图")
                    
                    # 调整图表大小
                    fig.update_layout(width=chart_width, height=chart_height)
                    
                    # 显示图表
                    st.plotly_chart(fig)
            else:
                st.write("查询结果为空")
        
        except Exception as e:
            st.error(f"查询执行错误: {e}")

else:
    st.info("请上传Excel文件")

# 数据导出功能已被移除
