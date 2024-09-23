import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 设置随机种子以确保可重复性
np.random.seed(42)

# 生成日期范围
start_date = datetime(2022, 1, 1)
end_date = datetime(2023, 12, 31)
date_range = pd.date_range(start=start_date, end=end_date, freq='D')

# 创建基础数据
n_rows = len(date_range) * 20  # 每天约20条记录

data = {
    'Date': np.random.choice(date_range, n_rows),
    'Product_ID': np.random.randint(1001, 1021, n_rows),
    'Category': np.random.choice(['Electronics', 'Clothing', 'Home & Garden', 'Books', 'Sports'], n_rows),
    'Brand': np.random.choice(['BrandA', 'BrandB', 'BrandC', 'BrandD', 'BrandE'], n_rows),
    'Price': np.random.uniform(10, 1000, n_rows).round(2),
    'Quantity': np.random.randint(1, 11, n_rows),
    'Customer_ID': np.random.randint(1, 1001, n_rows),
    'Age_Group': np.random.choice(['18-25', '26-35', '36-45', '46-55', '56+'], n_rows),
    'Gender': np.random.choice(['Male', 'Female', 'Other'], n_rows),
    'Location': np.random.choice(['North', 'South', 'East', 'West', 'Central'], n_rows),
    'Payment_Method': np.random.choice(['Credit Card', 'Debit Card', 'PayPal', 'Cash on Delivery'], n_rows),
    'Discount_Applied': np.random.choice([True, False], n_rows),
    'Delivery_Time': np.random.randint(1, 8, n_rows),
    'Rating': np.random.randint(1, 6, n_rows),
    'Review_Count': np.random.randint(0, 101, n_rows)
}

# 创建DataFrame
df = pd.DataFrame(data)

# 计算总销售额
df['Total_Sales'] = df['Price'] * df['Quantity']

# 添加季节信息
df['Season'] = df['Date'].dt.month.map({1:'Winter', 2:'Winter', 3:'Spring', 4:'Spring', 5:'Spring', 
                                        6:'Summer', 7:'Summer', 8:'Summer', 9:'Fall', 10:'Fall', 
                                        11:'Fall', 12:'Winter'})

# 添加是否为周末的标志
df['Is_Weekend'] = df['Date'].dt.dayofweek.isin([5, 6])

# 添加客户满意度（基于评分）
df['Customer_Satisfaction'] = df['Rating'] / 5

# 按日期排序
df = df.sort_values('Date')

# 保存为Excel文件
df.to_excel('ecommerce_data.xlsx', index=False)
print("Excel文件 'ecommerce_data.xlsx' 已生成。")